from io import BytesIO
from pypdf import PdfReader
import docx

def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".docx"):
        d = docx.Document(BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)

    raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx.")
