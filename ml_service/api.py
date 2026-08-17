import os
import io
import uuid
import time
import json
import asyncio
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pymysql
from contextlib import asynccontextmanager

# Load environment variables
import dotenv
dotenv.load_dotenv()

# MySQL Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "dermscan_db")

# In production, this would be an S3 bucket
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "../public/uploads/scans/")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure model info is in DB
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO models (model_version, model_architecture, model_checksum, preprocessing_version, status)
                VALUES ('v1.0.0', 'MobileNetV2', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'v1', 'PRODUCTION')
                ON DUPLICATE KEY UPDATE id=id;
            """)
        conn.close()
    except Exception as e:
        print(f"Failed to initialize DB on startup: {e}")
    yield
    # Shutdown

app = FastAPI(
    title="DermScan AI Inference Platform",
    description="Production-grade API for ML-driven skin lesion analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    timestamp: str

class ReadyResponse(BaseModel):
    status: str
    database: bool
    worker_queue: bool
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_version: str
    architecture: str
    status: str
    preprocessing_version: str

class ScanCreatedResponse(BaseModel):
    scan_id: str
    job_id: str
    status: str
    message: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )

@app.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    db_ok = False
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        conn.close()
        db_ok = True
    except:
        pass

    return ReadyResponse(
        status="ready" if db_ok else "unavailable",
        database=db_ok,
        worker_queue=db_ok,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM models WHERE status = 'PRODUCTION' ORDER BY created_at DESC LIMIT 1")
            model = cursor.fetchone()
        conn.close()
        
        if model:
            return ModelInfoResponse(
                model_version=model['model_version'],
                architecture=model['model_architecture'],
                status=model['status'],
                preprocessing_version=model['preprocessing_version']
            )
    except:
        pass
        
    return ModelInfoResponse(
        model_version="v1.0.0",
        architecture="MobileNetV2",
        status="PRODUCTION",
        preprocessing_version="v1"
    )

@app.post("/api/v1/scans", response_model=ScanCreatedResponse)
async def create_scan(
    request: Request,
    file: UploadFile = File(...),
    user_id: int = Form(...),
    body_location: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
):
    """
    Creates a scan record and processing job asynchronously.
    """
    conn = get_db_connection()
    try:
        # Check idempotency
        if idempotency_key:
            with conn.cursor() as cursor:
                cursor.execute("SELECT response_code, response_body FROM idempotency_records WHERE idempotency_key = %s", (idempotency_key,))
                record = cursor.fetchone()
                if record:
                    return JSONResponse(status_code=record['response_code'], content=json.loads(record['response_body']))

        # Validate file
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type.")

        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        
        # In a real system, we validate image corruption, dimensions, etc. here.
        
        # Save image (Object Storage Abstraction)
        filename = f"{uuid.uuid4()}_{file.filename}"
        object_key = f"uploads/scans/{filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as f:
            f.write(contents)

        scan_id = f"scan_{uuid.uuid4().hex}"
        job_id = f"job_{uuid.uuid4().hex}"

        # Transactional insert
        conn.begin()
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO skin_scans (user_id, scan_id, image_path, status, notes, body_location)
                VALUES (%s, %s, %s, 'QUEUED', %s, %s)
            """, (user_id, scan_id, object_key, notes, body_location))
            
            cursor.execute("""
                INSERT INTO processing_jobs (job_id, scan_id, status)
                VALUES (%s, %s, 'PENDING')
            """, (job_id, scan_id))
            
            # Audit log
            cursor.execute("""
                INSERT INTO audit_events (actor_id, actor_type, action, resource_type, resource_id, details)
                VALUES (%s, 'user', 'SCAN_CREATED', 'scan', %s, %s)
            """, (user_id, scan_id, json.dumps({"job_id": job_id})))
            
        conn.commit()

        response_data = {
            "scan_id": scan_id,
            "job_id": job_id,
            "status": "QUEUED",
            "message": "Scan processing job created successfully."
        }

        # Save idempotency
        if idempotency_key:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO idempotency_records (idempotency_key, path, response_code, response_body)
                    VALUES (%s, %s, %s, %s)
                """, (idempotency_key, request.url.path, 200, json.dumps(response_data)))

        return ScanCreatedResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
