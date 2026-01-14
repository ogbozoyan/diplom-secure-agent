from __future__ import annotations

import io
import logging

from docx import Document as DocxDocument
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader

from config import AppConfig

_log = logging.getLogger(__name__)

config = AppConfig()


# =========================
# Simple file parsing
# =========================
def check_file_limits( file: UploadFile ) -> None:
    # UploadFile doesn't always have size; simplest: read bytes and check
    data = file.file.read()
    file.file.seek(0)
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large: {file.filename}")
    return


def extract_text( file: UploadFile ) -> str:
    name = (file.filename or "").lower()
    data = file.file.read()
    file.file.seek(0)

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        parts = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(parts)

    if name.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    # txt/fallback
    try:
        return data.decode("utf-8", errors = "ignore")
    except Exception:
        return ""
