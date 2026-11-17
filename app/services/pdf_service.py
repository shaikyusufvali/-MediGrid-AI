import pdfplumber
import io


def extract_pdf_text(pdf_bytes):

    extracted_text = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

        if len(pdf.pages) > 30:
            raise ValueError("Maximum 30 PDF pages allowed")

        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text()

            if text:
                extracted_text.append(
                    f"\n--- Page {page_number} ---\n{text}"
                )

    result = "\n".join(extracted_text)

    if not result.strip():
        raise ValueError(
            "No readable text found. Scanned PDFs are not supported yet."
        )

    return result