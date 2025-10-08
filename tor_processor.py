# tor_processor.py
# Minimal, safe implementation so the app can boot
import re
from pdf_processor import PDFProcessor

class TORProcessor:
    def __init__(self):
        self.pdf = PDFProcessor()

    def extract_text(self, file_bytes):
        """Pass-through helper: get text from a PDF file."""
        return self.pdf.extract_text(file_bytes)

    def parse(self, file_bytes=None, text=None):
        """
        Very simple TOR parser: returns raw text and a naive split into sections.
        Works even if only text is provided.
        """
        if text is None and file_bytes is not None:
            text = self.extract_text(file_bytes) or ""
        text = text or ""
        # Naive sectioning by common TOR headings
        headings = r"(Background|Context|Objectives?|Scope|Activities|Deliverables?|Timeline|Budget|Qualifications?|Evaluation)"
        parts = re.split(rf"\n\s*{headings}\s*:?\s*\n", text, flags=re.I)
        return {"text": text, "sections": parts[:12]}

    def summarize(self, text, max_chars=1000):
        """Very basic fallback summarizer (keeps the app from crashing)."""
        text = text or ""
        return text if len(text) <= max_chars else text[:max_chars] + "…"

    # Safety net: any unknown method the app tries to call will just no-op
    def __getattr__(self, _name):
        def _fallback(*_args, **_kwargs):
            return None
        return _fallback
