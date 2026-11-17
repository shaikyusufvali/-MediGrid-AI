
import io
import pdfplumber


def load_pdf_document(pdf_bytes: bytes, filename: str):
    """
    Extract text from PDF page by page.

    Returns:
        List of dictionaries containing page text and metadata.
    """

    if not pdf_bytes:
        raise ValueError("PDF file is empty")

    documents = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

            if len(pdf.pages) > 30:
                raise ValueError("Maximum 30 PDF pages allowed")

            for page_number, page in enumerate(pdf.pages, start=1):

                text = page.extract_text()

                if text and text.strip():

                    documents.append({
                        "text": text.strip(),
                        "metadata": {
                            "source": filename,
                            "page": page_number
                        }
                    })

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            f"Unable to read PDF: {str(error)}"
        )

    if not documents:
        raise ValueError(
            "No readable text found in PDF"
        )

    return documents
