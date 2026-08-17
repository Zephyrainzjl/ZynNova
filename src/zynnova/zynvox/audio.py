"""Dependency-light audio I/O and integrity measurements."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import BackendUnavailableError, ConfigurationError


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        values = np.asarray(self.samples, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] < 1:
            raise ConfigurationError("audio samples must have shape [frames] or [frames, channels]")
        if self.sample_rate < 1:
            raise ConfigurationError("sample_rate must be positive")
        if not np.all(np.isfinite(values)):
            raise ConfigurationError("audio contains non-finite samples")
        object.__setattr__(self, "samples", np.clip(values, -1.0, 1.0))
        object.__setattr__(self, "sample_rate", int(self.sample_rate))

    @property
    def duration_s(self) -> float:
        return self.samples.shape[0] / self.sample_rate

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    def mono(self) -> AudioBuffer:
        if self.channels == 1:
            return self
        return AudioBuffer(self.samples.mean(axis=1, keepdims=True), self.sample_rate)


@dataclass(frozen=True, slots=True)
class AudioIntegrity:
    duration_s: float
    sample_rate: int
    channels: int
    peak_dbfs: float
    rms_dbfs: float
    clipping_fraction: float
    dc_offset: float


def read_audio(path: str | Path, *, mono: bool = False) -> AudioBuffer:
    """Read PCM WAV directly, delegating other formats to soundfile when installed."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        audio = _read_wave(source)
    except (wave.Error, EOFError):
        audio = _read_soundfile(source)
    return audio.mono() if mono else audio


def _read_wave(path: Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        frame_count = stream.getnframes()
        compression = stream.getcomptype()
        if compression != "NONE":
            raise wave.Error(f"unsupported WAV compression {compression!r}")
        raw = stream.readframes(frame_count)
    if sample_width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        integers = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        values = integers.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise wave.Error(f"unsupported PCM sample width {sample_width}")
    if values.size % channels:
        raise wave.Error("WAV frame/channel count is inconsistent")
    return AudioBuffer(values.reshape(-1, channels), sample_rate)


def _read_soundfile(path: Path) -> AudioBuffer:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise BackendUnavailableError(
            f"{path.suffix or 'audio'} decoding requires soundfile; "
            "install zynnova[zynnova-voice] or provide PCM WAV"
        ) from exc
    values, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    return AudioBuffer(values, int(sample_rate))


def write_wav(path: str | Path, audio: AudioBuffer) -> Path:
    """Write deterministic signed 16-bit PCM WAV."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = np.clip(audio.samples, -1.0, 1.0)
    pcm = np.rint(values * 32767.0).astype("<i2")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with wave.open(str(temporary), "wb") as stream:
        stream.setnchannels(audio.channels)
        stream.setsampwidth(2)
        stream.setframerate(audio.sample_rate)
        stream.writeframes(pcm.tobytes(order="C"))
    temporary.replace(target)
    return target


def resample(audio: AudioBuffer, sample_rate: int) -> AudioBuffer:
    """Resample using scipy polyphase when available, otherwise deterministic interpolation."""

    target_rate = int(sample_rate)
    if target_rate < 1:
        raise ConfigurationError("sample_rate must be positive")
    if target_rate == audio.sample_rate:
        return audio
    try:
        from scipy.signal import resample_poly
    except ImportError:
        old_count = audio.samples.shape[0]
        new_count = max(1, int(round(old_count * target_rate / audio.sample_rate)))
        old_x = np.arange(old_count, dtype=np.float64)
        new_x = np.linspace(0.0, max(0, old_count - 1), new_count)
        channels = [np.interp(new_x, old_x, audio.samples[:, index]) for index in range(audio.channels)]
        values = np.column_stack(channels).astype(np.float32)
    else:
        divisor = math.gcd(audio.sample_rate, target_rate)
        values = resample_poly(
            audio.samples,
            target_rate // divisor,
            audio.sample_rate // divisor,
            axis=0,
        ).astype(np.float32)
    return AudioBuffer(values, target_rate)


def peak_normalize(audio: AudioBuffer, target_dbfs: float = -1.0) -> AudioBuffer:
    target = float(10.0 ** (target_dbfs / 20.0))
    peak = float(np.max(np.abs(audio.samples)))
    if peak <= np.finfo(np.float32).eps:
        return audio
    gain = min(target / peak, 100.0)
    return AudioBuffer(audio.samples * gain, audio.sample_rate)


def audio_integrity(audio: AudioBuffer) -> AudioIntegrity:
    absolute = np.abs(audio.samples)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(audio.samples, dtype=np.float64))))
    epsilon = np.finfo(float).tiny
    return AudioIntegrity(
        duration_s=audio.duration_s,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        peak_dbfs=float(20.0 * np.log10(max(peak, epsilon))),
        rms_dbfs=float(20.0 * np.log10(max(rms, epsilon))),
        clipping_fraction=float(np.mean(absolute >= (32766.0 / 32767.0))),
        dc_offset=float(np.mean(audio.samples)),
    )


__all__ = [
    "AudioBuffer",
    "AudioIntegrity",
    "audio_integrity",
    "peak_normalize",
    "read_audio",
    "resample",
    "write_wav",
]
