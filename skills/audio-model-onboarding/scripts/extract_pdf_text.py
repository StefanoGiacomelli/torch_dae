"""Extract a bounded machine-readable text layer from one local PDF."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import emit_json
from pypdf import PdfReader

DEFAULT_MAX_FILE_SIZE_BYTES = 25_000_000
DEFAULT_MAX_PAGES = 200
DEFAULT_MAX_CHARACTERS = 1_000_000
HARD_MAX_FILE_SIZE_BYTES = 100_000_000
HARD_MAX_PAGES = 1_000
HARD_MAX_CHARACTERS = 5_000_000
MAX_METADATA_VALUE_CHARACTERS = 2_000


@dataclass(frozen=True)
class PdfExtractionFailure(Exception):
    """Expected, sanitized PDF extraction failure."""

    status: str
    message: str
    limit_name: str | None = None
    observed: int | None = None
    maximum: int | None = None


def extract_pdf_text(
    path: Path,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
) -> dict[str, Any]:
    """Extract bounded text and metadata from a local PDF without OCR or external processes."""

    _validate_limit("max_file_size_bytes", max_file_size_bytes, HARD_MAX_FILE_SIZE_BYTES)
    _validate_limit("max_pages", max_pages, HARD_MAX_PAGES)
    _validate_limit("max_characters", max_characters, HARD_MAX_CHARACTERS)
    source_text = str(path)
    if source_text.startswith(("http://", "https://", "ftp://", "file://")):
        raise PdfExtractionFailure("non_local_input", "input must be a local filesystem path")
    if path.is_symlink():
        raise PdfExtractionFailure("unsupported_input", "symlinked PDF paths are not accepted")
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise PdfExtractionFailure("missing_input", "local PDF does not exist") from exc
    except (OSError, PermissionError) as exc:
        raise PdfExtractionFailure("unreadable_input", "local PDF cannot be inspected") from exc
    if not path.is_file():
        raise PdfExtractionFailure("unsupported_input", "input is not a regular file")
    if stat.st_size > max_file_size_bytes:
        raise PdfExtractionFailure(
            "limit_exceeded",
            "PDF exceeds the configured file-size limit",
            "file_size_bytes",
            stat.st_size,
            max_file_size_bytes,
        )
    try:
        with path.open("rb") as stream:
            signature = stream.read(5)
    except (OSError, PermissionError) as exc:
        raise PdfExtractionFailure("unreadable_input", "local PDF cannot be read") from exc
    if signature != b"%PDF-":
        raise PdfExtractionFailure("not_pdf", "input does not begin with a PDF signature")

    try:
        reader = PdfReader(path, strict=True)
    except Exception as exc:
        raise PdfExtractionFailure(
            "parser_failure",
            f"pypdf could not parse the input ({type(exc).__name__})",
        ) from exc
    try:
        encrypted = reader.is_encrypted
    except Exception as exc:
        raise PdfExtractionFailure(
            "parser_failure",
            f"pypdf could not inspect encryption state ({type(exc).__name__})",
        ) from exc
    if encrypted:
        raise PdfExtractionFailure(
            "encrypted",
            "encrypted PDFs are not decrypted or inspected",
        )
    try:
        page_count = len(reader.pages)
    except Exception as exc:
        raise PdfExtractionFailure(
            "parser_failure",
            f"pypdf could not enumerate pages ({type(exc).__name__})",
        ) from exc
    if page_count > max_pages:
        raise PdfExtractionFailure(
            "limit_exceeded",
            "PDF exceeds the configured page-count limit",
            "page_count",
            page_count,
            max_pages,
        )

    pages: list[dict[str, object]] = []
    character_count = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise PdfExtractionFailure(
                "parser_failure",
                f"pypdf could not extract page {page_number} ({type(exc).__name__})",
            ) from exc
        character_count += len(text)
        if character_count > max_characters:
            raise PdfExtractionFailure(
                "limit_exceeded",
                "extracted text exceeds the configured character limit",
                "extracted_characters",
                character_count,
                max_characters,
            )
        pages.append({"page": page_number, "text": text})

    has_text_layer = any(str(page["text"]).strip() for page in pages)
    return {
        "status": "ok" if has_text_layer else "no_text_layer",
        "input_name": path.name,
        "file_size_bytes": stat.st_size,
        "page_count": page_count,
        "extracted_character_count": character_count,
        "has_text_layer": has_text_layer,
        "ocr_provided": False,
        "metadata": _safe_metadata(reader),
        "pages": pages,
    }


def render_text(payload: dict[str, Any]) -> str:
    """Render deterministic UTF-8 text with explicit page boundaries."""

    lines = [
        f"Status: {payload['status']}",
        f"Input: {payload['input_name']}",
        f"Pages: {payload['page_count']}",
        f"Extracted characters: {payload['extracted_character_count']}",
        "OCR: not provided",
        "Metadata:",
    ]
    metadata = payload["metadata"]
    if isinstance(metadata, dict) and metadata:
        lines.extend(f"- {key}: {metadata[key]}" for key in sorted(metadata))
    else:
        lines.append("- none")
    for page in payload["pages"]:
        if not isinstance(page, dict):
            continue
        lines.extend(["", f"--- page {page['page']} ---", str(page["text"])])
    return "\n".join(lines).rstrip() + "\n"


def _safe_metadata(reader: PdfReader) -> dict[str, str]:
    try:
        metadata = reader.metadata
    except Exception:
        return {}
    if metadata is None:
        return {}
    safe: dict[str, str] = {}
    for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
        if value is None:
            continue
        rendered = str(value)
        safe[str(key)] = rendered[:MAX_METADATA_VALUE_CHARACTERS]
    return safe


def _validate_limit(name: str, value: int, hard_maximum: int) -> None:
    if value < 1 or value > hard_maximum:
        raise PdfExtractionFailure(
            "invalid_limit",
            f"{name} must be between 1 and {hard_maximum}",
            name,
            value,
            hard_maximum,
        )


def _failure_payload(exc: PdfExtractionFailure) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": exc.status,
        "message": exc.message,
        "has_text_layer": False,
        "ocr_provided": False,
    }
    if exc.limit_name is not None:
        payload["limit"] = {
            "name": exc.limit_name,
            "observed": exc.observed,
            "maximum": exc.maximum,
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="local PDF path; URLs are not accepted")
    parser.add_argument("--json", action="store_true", help="emit deterministic JSON")
    parser.add_argument(
        "--max-file-size-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE_BYTES,
        help=f"maximum input bytes (hard maximum {HARD_MAX_FILE_SIZE_BYTES})",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f"maximum pages (hard maximum {HARD_MAX_PAGES})",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=DEFAULT_MAX_CHARACTERS,
        help=f"maximum extracted characters (hard maximum {HARD_MAX_CHARACTERS})",
    )
    args = parser.parse_args()
    try:
        payload = extract_pdf_text(
            args.pdf,
            max_file_size_bytes=args.max_file_size_bytes,
            max_pages=args.max_pages,
            max_characters=args.max_characters,
        )
    except PdfExtractionFailure as exc:
        failure = _failure_payload(exc)
        if args.json:
            emit_json(failure)
        else:
            print(f"error: {exc.status}: {exc.message}")
        return 2
    if args.json:
        emit_json(payload)
    else:
        print(render_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
