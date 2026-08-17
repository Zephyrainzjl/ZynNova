from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def image_factory():
    def create(path: Path, *, size: int = 32, style: bool = False) -> Path:
        from PIL import Image, ImageDraw

        if style:
            image = Image.new("RGB", (size, size), (25, 45, 125))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, size // 2, size), fill=(220, 115, 40))
            draw.ellipse((size // 3, size // 4, size - 2, size - 2), fill=(40, 210, 145))
        else:
            image = Image.new("RGB", (size, size), (245, 245, 245))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle(
                (size // 5, size // 6, size - size // 5, size - size // 7),
                radius=max(2, size // 8),
                fill=(35, 85, 180),
            )
            draw.ellipse(
                (size // 3, size // 4, 2 * size // 3, 3 * size // 4),
                fill=(225, 70, 45),
            )
        image.save(path)
        return path

    return create


@pytest.fixture
def wav_factory():
    def create(
        path: Path,
        *,
        frequency_hz: float = 220.0,
        duration_s: float = 0.16,
        sample_rate: int = 16_000,
    ) -> Path:
        time = np.arange(int(round(duration_s * sample_rate)), dtype=np.float64) / sample_rate
        signal = 0.18 * np.sin(2.0 * np.pi * frequency_hz * time)
        pcm = np.rint(np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        return path

    return create
