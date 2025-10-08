import PyPDF2
import io

class PDFProcessor:
    def __init__(self):
        pass

    def extract_text(self, file_bytes):
        """Extracts text from uploaded PDF file"""
        pdf_text = ""
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                pdf_text += page.extract_text() or ""
        except Exception as e:
            pdf_text = f"Error reading PDF: {e}"
        return pdf_text
