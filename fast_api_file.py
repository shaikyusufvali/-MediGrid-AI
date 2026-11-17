from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Form
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from typing import List, Optional

from app.rag.rag_pipeline import (
    ingest_pdf,
    ask_rag
)

import io
import json
import re
import traceback

import Extracting_prescription_data
import Critical_Warinings
import SQLLITE3_DataBase

from AI_assistant_logic import MedicalAssistant


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

api = FastAPI(

    title="MediGrid AI",

    description="AI Powered Healthcare Assistant",

    version="3.0.0"

)


# ==========================================
# 2. STATIC FILES
# ==========================================

api.mount(

    "/static",

    StaticFiles(directory="static"),

    name="static"

)


@api.get("/")
async def read_root():

    return FileResponse("static/test1.html")


# ==========================================
# 3. CORS CONFIGURATION
# ==========================================

api.add_middleware(

    CORSMiddleware,

    allow_origins=[

        "http://127.0.0.1:8000",

        "http://localhost:8000",

        "http://localhost:5173"

    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


# ==========================================
# 4. AI ASSISTANT INITIALIZATION
# ==========================================

ai_bot = MedicalAssistant()


# ==========================================
# 5. PYDANTIC MODELS
# ==========================================

class MedicationItem(BaseModel):

    medications: str

    Dosage: str

    Frequency: str

    Duration: str

    Map_link: str


class PrescriptionRequest(BaseModel):

    Prescription_info: List[MedicationItem]


class ChatRequest(BaseModel):

    message: str


# ==========================================
# 6. PRESCRIPTION DATA EXTRACTION
# ==========================================

@api.post("/data_extraction")
async def prescription_dataExtraction(

    image: UploadFile = File(...),

    user_location: Optional[str] = Form(None)

):

    try:

        content = await image.read()

        if not content:

            raise HTTPException(

                status_code=400,

                detail="Empty image uploaded"

            )

        img_byte = io.BytesIO(content)

        loc_data_dict = None

        if user_location:

            try:

                loc_data_dict = json.loads(
                    user_location
                )

                print(
                    "Received Location:",
                    loc_data_dict
                )

            except json.JSONDecodeError:

                raise HTTPException(

                    status_code=400,

                    detail="Invalid location JSON"

                )

        obj = (
            Extracting_prescription_data
            .Handwritting_Extraction()
        )

        result = obj.extracting_presc_data(

            img_byte,

            loc_data_dict

        )

        response = ""

        for chunk in result:

            if chunk.parts:

                response += chunk.text

        match = re.search(

            r"(\{.*\})",

            response,

            re.DOTALL

        )

        clean_json = (
            match.group(1)
            if match
            else response
        )

        data = json.loads(clean_json)

        return data

    except HTTPException:

        raise

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(

            status_code=500,

            detail=f"Prescription extraction failed: {str(e)}"

        )


# ==========================================
# 7. CRITICAL WARNINGS
# ==========================================

@api.post("/critical_warnings")
async def critical_warnings_handler(

    extracted_data: PrescriptionRequest

):

    try:

        warn_obj = Critical_Warinings.warnings()

        warn_response = (
            warn_obj.analyzing_critical_warnings(
                extracted_data.model_dump()
            )
        )

        return warn_response

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ==========================================
# 8. SAVE MEDICINES TO DATABASE
# ==========================================

@api.post("/post_into_db")
def save_to_sql(payload: dict):

    try:

        sql_obj = SQLLITE3_DataBase.medi_data_base()

        result = sql_obj.save_to_db(payload)

        return result

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ==========================================
# 9. GET SAVED MEDICINES
# ==========================================

@api.get("/get_Saved_data")
def get_saved_details():

    try:

        sql_obj = SQLLITE3_DataBase.medi_data_base()

        result = sql_obj.display_table()

        return result.to_dict(
            orient="records"
        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ==========================================
# 10. AI CHATBOT WITH ADVANCED RAG
# ==========================================

@api.post("/chat")
async def chat_endpoint(req: ChatRequest):

    try:

        if not req.message.strip():

            raise HTTPException(

                status_code=400,

                detail="Message cannot be empty"

            )

        # Step 1: Retrieve relevant PDF information

        rag_result = ask_rag(

            question=req.message,

            top_k=3

        )

        retrieved_context = rag_result.get(
            "context", []
        )

        # Step 2: Send question and RAG context to Gemini

        response_text = ai_bot.process_chat(

            req.message,

            session_key="default_user",

            rag_context=retrieved_context

        )

        # Step 3: Prepare source references

        sources = []

        seen_sources = set()

        for item in retrieved_context:

            source = item.get(
                "source", "Unknown"
            )

            page = item.get(
                "page", "Unknown"
            )

            source_key = (source, page)

            if source_key not in seen_sources:

                sources.append({

                    "filename": source,

                    "page": page

                })

                seen_sources.add(source_key)

        return {

            "response": response_text,

            "rag_status": rag_result.get(
                "status"
            ),

            "sources": sources

        }

    except HTTPException:

        raise

    except Exception as e:

        print("Chat Error:", e)

        traceback.print_exc()

        raise HTTPException(

            status_code=500,

            detail="AI Chat processing failed"

        )


# ==========================================
# 11. PDF KNOWLEDGE BASE UPLOAD WITH RAG
# ==========================================

@api.post("/upload_knowledge_pdf")
async def upload_knowledge_pdf(

    file: UploadFile = File(...)

):

    if not file.filename or not file.filename.lower().endswith(".pdf"):

        raise HTTPException(

            status_code=400,

            detail="Please upload a PDF file"

        )

    try:

        # Read PDF with 10 MB limit

        content = await file.read(
            10 * 1024 * 1024 + 1
        )

        if len(content) > 10 * 1024 * 1024:

            raise HTTPException(

                status_code=413,

                detail="PDF size must be below 10 MB"

            )

        # Validate PDF signature

        if not content.startswith(b"%PDF-"):

            raise HTTPException(

                status_code=400,

                detail="Invalid PDF file"

            )

        # Process PDF through RAG pipeline

        result = ingest_pdf(

            pdf_bytes=content,

            filename=file.filename

        )

        # Reset existing chat sessions

        ai_bot.chat_sessions.clear()

        return {

            "status": "success",

            "filename": file.filename,

            "message": "PDF successfully added to RAG Knowledge Base",

            **result

        }

    except HTTPException:

        raise

    except ValueError as e:

        raise HTTPException(

            status_code=400,

            detail=str(e)

        )

    except Exception as e:

        print("PDF Upload Error:", e)

        traceback.print_exc()

        raise HTTPException(

            status_code=500,

            detail="PDF processing failed"

        )

    finally:

        await file.close()


# ==========================================
# 12. RAG KNOWLEDGE BASE STATUS
# ==========================================

@api.get("/knowledge_status")
async def knowledge_status():

    from app.rag.vector_store import get_collection_count

    return {

        "application": "MediGrid AI",

        "rag_status": "active",

        "total_stored_chunks": get_collection_count()

    }


# ==========================================
# 13. HEALTH CHECK
# ==========================================

@api.get("/health")
async def health_check():

    return {

        "application": "MediGrid AI",

        "status": "running",

        "version": "3.0.0"

    }