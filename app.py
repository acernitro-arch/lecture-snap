"""LectureSnap - local slide-frame extractor for permitted YouTube lectures."""
from __future__ import annotations

import shutil
import sys
import uuid
import os
from pathlib import Path

import cv2
import numpy as np
import yt_dlp
from flask import Flask, jsonify, render_template, request, send_from_directory

# The packaged local runtime includes ReportLab; the app's video environment is
# intentionally separate, so expose it when this app is launched from its bat file.
BUNDLED_SITE_PACKAGES = Path(r"C:\Users\thaka\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages")
if BUNDLED_SITE_PACKAGES.exists():
    sys.path.append(str(BUNDLED_SITE_PACKAGES))
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

BASE = Path(__file__).resolve().parent
JOBS = BASE / "jobs"
JOBS.mkdir(exist_ok=True)

app = Flask(__name__)


def is_youtube_url(value: str) -> bool:
    return "youtube.com/" in value or "youtu.be/" in value


def download_video(url: str, destination: Path) -> Path:
    options = {
        "format": "best[ext=mp4][vcodec!=none][acodec!=none]/best[ext=mp4]/best",
        "outtmpl": str(destination / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.extract_info(url, download=True)
    videos = sorted(destination.glob("source.*"))
    if not videos:
        raise RuntimeError("The video could not be downloaded.")
    return videos[0]


def visual_change(previous: np.ndarray, current: np.ndarray) -> float:
    """Return the fraction of the screen that materially changed."""
    previous = cv2.resize(previous, (320, 180))
    current = cv2.resize(current, (320, 180))
    previous = cv2.GaussianBlur(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    current = cv2.GaussianBlur(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    difference = cv2.absdiff(previous, current)
    return float(np.mean(difference > 28))


def ink_density(frame: np.ndarray) -> float:
    """Approximate how much writing/diagram detail is visible on a board."""
    height, width = frame.shape[:2]
    # Exclude the bottom/outer edges where a lecturer commonly appears. This
    # makes a person walking across the board much less likely to look like a
    # board clear.
    frame = frame[int(height * 0.04):int(height * 0.82), int(width * 0.06):int(width * 0.98)]
    small = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 65, 145)
    return float(np.mean(edges > 0))


def board_information_score(frame: np.ndarray) -> float:
    """Favor frames containing the most written board content, not an erasure."""
    height, width = frame.shape[:2]
    # Ignore the very bottom strip (where lecturers commonly stand) while keeping
    # nearly the entire board. Thin handwriting creates a useful edge signal.
    board = frame[int(height * 0.04):int(height * 0.82), int(width * 0.06):int(width * 0.98)]
    gray = cv2.GaussianBlur(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, 55, 135)
    return float(np.mean(edges > 0))


def board_content_change(previous: np.ndarray, current: np.ndarray) -> float:
    """Compare the least-obstructed upper-right board area between keyframes."""
    height, width = previous.shape[:2]
    region = (slice(int(height * 0.04), int(height * 0.52)), slice(int(width * 0.28), int(width * 0.98)))
    return visual_change(previous[region], current[region])


def clean_board(final: np.ndarray, recent_frames: list[np.ndarray]) -> np.ndarray:
    """Fill moving presenter-shaped regions with nearby visible board pixels.

    This is deliberately conservative: only large, moving regions are replaced.
    If a teacher blocks the same writing for the whole window, the image is kept
    unchanged rather than guessing what was behind them.
    """
    if len(recent_frames) < 3:
        return final
    stack = np.stack(recent_frames)
    # A white/digital board is normally brighter than a lecturer. Taking the
    # brightest nearby pixel avoids the face-shaped ghosts a median can create.
    # It only works when the teacher has moved enough to reveal that board area.
    background = np.max(stack, axis=0).astype(np.uint8)
    difference = cv2.cvtColor(cv2.absdiff(final, background), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(difference, 38, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean_mask = np.zeros_like(mask)
    min_area = final.shape[0] * final.shape[1] * 0.004
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == label] = 255
    # Use a solid replacement mask, not a translucent blend. Blending is what
    # created the unwanted see-through outline around the teacher.
    clean_mask = cv2.dilate(clean_mask, np.ones((5, 5), np.uint8), iterations=1)
    return np.where(clean_mask[..., None] > 0, background, final)


def add_final_frame(frame: np.ndarray, history: list[np.ndarray], output: Path, saved: list[Path], remove_presenter: bool) -> None:
    candidate = clean_board(frame, history) if remove_presenter else frame
    # Empty or mostly erased boards are not useful revision notes.
    if board_information_score(candidate) < 0.035:
        return
    # Consecutive frames with the same board are merged; retain the fuller one.
    if saved:
        existing = cv2.imread(str(saved[-1]))
        if existing is not None and board_content_change(existing, candidate) < 0.055:
            if board_information_score(candidate) > board_information_score(existing):
                cv2.imwrite(str(saved[-1]), candidate, [cv2.IMWRITE_JPEG_QUALITY, 94])
            return
    saved.append(save_image(candidate, output, len(saved) + 1))


def extract_final_frames(video_path: Path, output: Path, every_seconds: float, threshold: float, mode: str, remove_presenter: bool) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps
    if duration <= 0:
        raise RuntimeError("LectureSnap could not read this video.")

    # A new slide normally alters a large fraction of the screen. Smaller changes
    # (writing, underlining, cursor motion) update the saved final state instead.
    slide_break = max(0.10, min(0.55, threshold))
    prior = None
    best_for_slide = None
    peak_ink = 0.0
    best_information = 0.0
    recent_frames: list[np.ndarray] = []
    section_started_at = 0.0
    min_board_seconds = 8.0
    saved: list[Path] = []
    timestamp = 0.0
    next_sample_at = 0.0
    frame_number = 0

    # Decode sequentially rather than seeking once per sample. Seeking through a
    # 25-minute YouTube file is extremely slow and can miss keyframes.
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_number / fps
        frame_number += 1
        if timestamp + 0.0001 < next_sample_at:
            continue
        next_sample_at += every_seconds
        current_ink = ink_density(frame)
        if prior is None:
            prior = frame
            best_for_slide = frame
            peak_ink = current_ink
            best_information = board_information_score(frame)
            recent_frames = [frame]
            section_started_at = timestamp
        else:
            changed = visual_change(prior, frame)
            # Whiteboard lectures often change gradually while the teacher writes.
            # A substantial fall in visible ink is a strong signal that the board
            # has been cleared, so retain the previous completed board state.
            board_cleared = (
                mode == "whiteboard"
                and peak_ink > 0.003
                and current_ink < peak_ink * 0.55
                and (peak_ink - current_ink) > 0.003
            )
            # In whiteboard mode, additions can look like a large visual change.
            # Only a genuine clear/reset ends a board section. Slide mode retains
            # the conventional full-screen slide-change detector.
            is_new_section = board_cleared if mode == "whiteboard" else changed >= slide_break
            if is_new_section:
                enough_content = peak_ink > 0.003 and (timestamp - section_started_at) >= min_board_seconds
                if best_for_slide is not None and enough_content:
                    add_final_frame(best_for_slide, recent_frames, output, saved, remove_presenter and mode == "whiteboard")
                best_for_slide = frame
                peak_ink = current_ink
                best_information = board_information_score(frame)
                recent_frames = [frame]
                section_started_at = timestamp
            else:
                # As the teacher writes, keep the fullest board state. During a
                # gradual wipe, later frames have less information and are ignored.
                information = board_information_score(frame)
                if mode != "whiteboard" or information >= best_information:
                    best_for_slide = frame
                    best_information = information
                peak_ink = max(peak_ink, current_ink)
                recent_frames.append(frame)
                # Only use the final 5 seconds to reconstruct a board obscured by a person.
                while len(recent_frames) > max(3, int(5 / every_seconds) + 1):
                    recent_frames.pop(0)
            prior = frame
    if best_for_slide is not None and (mode != "whiteboard" or peak_ink > 0.003):
        add_final_frame(best_for_slide, recent_frames, output, saved, remove_presenter and mode == "whiteboard")
    capture.release()
    return saved


def save_image(frame: np.ndarray, output: Path, number: int) -> Path:
    path = output / f"slide-{number:03d}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return path


def create_pdf(folder: Path, images: list[Path]) -> Path:
    """Create a clean one-board-per-page revision PDF."""
    pdf_path = folder / "lecturesnap-board-notes.pdf"
    page_width, page_height = landscape(A4)
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_width, page_height))
    for page_number, image_path in enumerate(images, 1):
        image = ImageReader(str(image_path))
        image_width, image_height = image.getSize()
        max_width, max_height = page_width - 64, page_height - 76
        scale = min(max_width / image_width, max_height / image_height)
        draw_width, draw_height = image_width * scale, image_height * scale
        x = (page_width - draw_width) / 2
        y = 38 + (max_height - draw_height) / 2
        pdf.setFillColorRGB(0.09, 0.13, 0.20)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(32, page_height - 24, "LectureSnap - Board Notes")
        pdf.setFont("Helvetica", 9)
        pdf.drawRightString(page_width - 32, page_height - 24, f"Page {page_number} of {len(images)}")
        pdf.drawImage(image, x, y, draw_width, draw_height, preserveAspectRatio=True, mask="auto")
        pdf.showPage()
    pdf.save()
    return pdf_path


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/extract")
def extract():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not is_youtube_url(url):
        return jsonify(error="Please paste a valid YouTube lecture link."), 400
    sample_every = float(data.get("sample_every", 2))
    sensitivity = float(data.get("sensitivity", 0.24))
    mode = str(data.get("mode", "whiteboard"))
    remove_presenter = bool(data.get("remove_presenter", False))
    if not 0.5 <= sample_every <= 10 or not 0.10 <= sensitivity <= 0.55 or mode not in {"whiteboard", "slides"}:
        return jsonify(error="Those extraction settings are not valid."), 400

    job_id = uuid.uuid4().hex[:12]
    folder = JOBS / job_id
    folder.mkdir()
    try:
        video = download_video(url, folder)
        images = extract_final_frames(video, folder, sample_every, sensitivity, mode, remove_presenter)
        if not images:
            raise RuntimeError("No slide frames were detected.")
        pdf = create_pdf(folder, images)
        return jsonify(
            job_id=job_id,
            count=len(images),
            pdf=f"/files/{job_id}/{pdf.name}",
            images=[f"/files/{job_id}/{image.name}" for image in images],
        )
    except Exception as error:
        shutil.rmtree(folder, ignore_errors=True)
        return jsonify(error=f"Could not process this lecture: {error}"), 422


@app.get("/files/<job_id>/<path:filename>")
def files(job_id: str, filename: str):
    return send_from_directory(JOBS / job_id, filename, as_attachment=filename.endswith(".zip"))


if __name__ == "__main__":
    # A single stable local process is more reliable when launched from a .bat
    # file than Flask's development reloader, which spawns a second process.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False)
