"""
detect_slides.py
----------------
Extracts frames from a video whenever the content differs significantly
from the last saved frame, using SSIM comparison.

Usage:
    python detect_slides.py --video meeting.mp4
    python detect_slides.py --video meeting.mp4 --output my_slides/ --threshold 0.90
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
from skimage.metrics import structural_similarity as ssim


# ── Tunable ───────────────────────────────────────────────────────────────────
SSIM_THRESHOLD = 0.92   # below this → frame is different enough to save
SAMPLE_FPS     = 1      # how many frames to check per second of video
# ─────────────────────────────────────────────────────────────────────────────


def compare_frames(frame_a, frame_b):
    """Return SSIM score between two BGR frames. 1.0 = identical, 0.0 = completely different."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # Crop out bottom 10% to ignore cursor/taskbar noise
    h = gray_a.shape[0]
    gray_a = gray_a[:int(h * 0.90), :]
    gray_b = gray_b[:int(h * 0.90), :]

    score, _ = ssim(gray_a, gray_b, full=True)
    return score


def extract_slide_frames(video_path, output_dir, threshold=SSIM_THRESHOLD, sample_fps=SAMPLE_FPS):
    """
    Walk through the video at sample_fps, comparing each frame to the last
    saved frame. Save a new image whenever they differ enough.

    Returns a list of dicts: [{slide_num, timestamp, filepath, ssim_score}]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps
    step = int(video_fps / sample_fps)  # how many frames to skip between checks

    print(f"Video: {video_path}")
    print(f"  FPS: {video_fps:.1f} | Duration: {duration_sec:.1f}s | Checking every {step} frames")
    print(f"  SSIM threshold: {threshold} | Output: {output_dir}\n")

    last_saved_frame = None
    slide_events = []
    slide_num = 0
    frame_idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / video_fps

        if last_saved_frame is None:
            # Always save the very first frame
            score = 0.0
            should_save = True
        else:
            score = compare_frames(last_saved_frame, frame)
            should_save = score < threshold

        if should_save:
            slide_num += 1
            filename = f"slide_{slide_num:04d}_{int(timestamp):05d}s.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            slide_events.append({
                "slide_num": slide_num,
                "timestamp": round(timestamp, 2),
                "filepath": filepath,
                "ssim_score": round(score, 4),
            })

            last_saved_frame = frame.copy()
            print(f"  [{slide_num:>4}] t={timestamp:>7.1f}s  ssim={score:.4f}  → {filename}")

        frame_idx += step

    cap.release()

    # Save timeline JSON alongside the images
    timeline_path = os.path.join(output_dir, "timeline.json")
    with open(timeline_path, "w") as f:
        json.dump(slide_events, f, indent=2)

    print(f"\nDone. {slide_num} slide frames saved to '{output_dir}/'")
    print(f"Timeline saved to '{timeline_path}'")
    return slide_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract unique slide frames from a meeting recording.")
    parser.add_argument("--video",     required=True,           help="Path to the video file (mp4, mkv, etc.)")
    parser.add_argument("--output",    default="slides",        help="Output directory for saved frames (default: slides/)")
    parser.add_argument("--threshold", type=float, default=SSIM_THRESHOLD,
                        help=f"SSIM threshold — frames below this are saved (default: {SSIM_THRESHOLD})")
    parser.add_argument("--fps",       type=float, default=SAMPLE_FPS,
                        help=f"Frames to check per second of video (default: {SAMPLE_FPS})")
    args = parser.parse_args()

    extract_slide_frames(
        video_path=args.video,
        output_dir=args.output,
        threshold=args.threshold,
        sample_fps=args.fps,
    )
    # python videoSegmentation.py --video presentation.mp4 --output slides/