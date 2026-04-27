import io
from docx import Document
from fastapi import UploadFile
from pypdf import PdfReader

async def extract_text_from_file(file: UploadFile) -> str:
    """
    Extracts text from PDF or DOCX files.
    """
    content = await file.read()
    filename = file.filename.lower()
    
    # Seek back to 0 in case we need to read it again (though we shouldn't)
    await file.seek(0)
    
    if filename.endswith(".pdf"):
        return _parse_pdf(content)
    elif filename.endswith(".docx"):
        return _parse_docx(content)
    elif filename.endswith(".txt"):
        return content.decode("utf-8")
    else:
        # Fallback to plain text decode if possible
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported file type: {filename}")

def _parse_pdf(content: bytes) -> str:
    text_parts: list[str] = []
    pdf = PdfReader(io.BytesIO(content))
    for page in pdf.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def _parse_docx(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    return "\n".join([para.text for para in doc.paragraphs])
