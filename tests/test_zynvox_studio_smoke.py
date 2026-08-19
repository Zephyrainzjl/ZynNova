from __future__ import annotations

from pathlib import Path

from zynnova.zynvox import ConsentBasis, ConsentRecord
from zynnova.zynvox.studio import GenerationRequest, GenerationResult, ZynVoxStudio


class DummyEngine:
    name = "dummy"

    def synthesize(self, request, profile, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFFdummyWAVE")
        return GenerationResult(output, self.name, profile.model, 0.01)

    def convert(self, source, profile, output, **options):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(Path(source).read_bytes())
        return GenerationResult(output, self.name, profile.model, 0.01)


def test_studio_enroll_and_synthesize(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"RIFFreferenceWAVE")
    studio = ZynVoxStudio(tmp_path / "workspace", engine=DummyEngine())
    consent = ConsentRecord(True, ConsentBasis.SELF, "unit test voice")
    studio.enroll_voice("unit", reference, consent, reference_text="hello", language="en")
    assert studio.list_voices() == ("unit",)
    result = studio.synthesize(GenerationRequest("hello world", "unit", language="en"))
    assert result.audio.is_file()
    assert result.engine == "dummy"
