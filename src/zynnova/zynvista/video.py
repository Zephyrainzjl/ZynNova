"""Video sampling for image-only metric reconstruction backends."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def extract_video_frames(
    video: str | Path,
    directory: str | Path,
    *,
    sample_fps: float = 2.0,
    maximum_frames: int = 96,
    image_quality: int = 2,
) -> tuple[Path, ...]:
    """Sample a video into ordered RGB frames.

    FFmpeg is preferred because it supports long videos without loading them into
    memory.  ``imageio`` is a portable fallback for notebook environments.
    """

    source = Path(video).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive")
    if maximum_frames < 1:
        raise ValueError("maximum_frames must be positive")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        pattern = target / "frame_%06d.jpg"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"fps={sample_fps:.12g}",
            "-frames:v",
            str(int(maximum_frames)),
            "-q:v",
            str(int(image_quality)),
            str(pattern),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode == 0:
            frames = tuple(sorted(target.glob("frame_*.jpg")))
            if frames:
                return frames

    try:
        import imageio.v3 as iio
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "video sampling requires ffmpeg or imageio+pillow; install zynnova[zynnova-scene]"
        ) from exc

    metadata = {}
    try:
        metadata = dict(iio.immeta(source))
    except Exception:
        metadata = {}
    source_fps = float(metadata.get("fps", 0.0) or 0.0)
    stride = max(1, int(round(source_fps / sample_fps))) if source_fps > 0.0 else 1
    frames: list[Path] = []
    for index, frame in enumerate(iio.imiter(source)):
        if index % stride:
            continue
        path = target / f"frame_{len(frames) + 1:06d}.jpg"
        Image.fromarray(frame).convert("RGB").save(path, quality=95)
        frames.append(path)
        if len(frames) >= maximum_frames:
            break
    if not frames:
        raise RuntimeError(f"no frames could be decoded from {source}")
    return tuple(frames)


__all__ = ["extract_video_frames"]
