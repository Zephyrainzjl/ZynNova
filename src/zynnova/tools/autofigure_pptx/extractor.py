from __future__ import annotations

import re
from pathlib import Path

from .exceptions import MissingDependencyError


_METHOD_HEADINGS = (
    "method", "methods", "methodology", "materials and methods", "experimental", "approach",
    "方法", "研究方法", "材料与方法", "技术路线",
)


class PaperMethodExtractor:
    """Extract method text from PDF, Markdown or plain-text manuscripts."""

    def extract(self, paper_path: str | Path) -> str:
        path = Path(paper_path).expanduser().resolve()
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = self._pdf_text(path)
        elif suffix in {".md", ".markdown", ".txt", ".rst"}:
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"unsupported paper format {suffix!r}")
        return self.extract_methodology(text)

    def _pdf_text(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise MissingDependencyError("pypdf is required for paper-to-figure PDF extraction") from exc
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    def extract_methodology(self, text: str) -> str:
        lines = text.splitlines()
        start = None
        for index, line in enumerate(lines):
            normalized = re.sub(r"[^a-zA-Z\u4e00-\u9fff ]", "", line).strip().lower()
            if any(normalized == heading or normalized.startswith(heading + " ") for heading in _METHOD_HEADINGS):
                start = index + 1
                break
        if start is None:
            return text.strip()
        selected: list[str] = []
        for line in lines[start:]:
            stripped = line.strip()
            if selected and re.match(r"^(?:#{1,3}\s+|\d+[.]\s+)?(?:results?|discussion|conclusion|结果|讨论|结论)\b", stripped, re.I):
                break
            selected.append(line)
        result = "\n".join(selected).strip()
        return result or text.strip()
