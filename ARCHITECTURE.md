# DermScan AI - Target Architecture

## Architectural Overview
DermScan AI has been systematically refactored from a tightly coupled synchronous application into a robust, event-driven, service-oriented platform. The core goal of this architecture is to ensure high reliability during ML inference, provide clear boundaries between frontend logic and backend processing, and establish a framework for clinical auditability.

## System Components

### 1. Web Application Gateway (PHP 8.x / Twig)
The PHP layer serves as the primary gateway and presentation layer. It is responsible for:
- User Authentication & Session Management
- Business Logic Validation (Input sanitation, Idempotency checks)
- File Uploads & Local Storage Abstraction
- Job Dispatching (Inserting to `processing_jobs`)

### 2. ML Inference Platform (Python / FastAPI)
A dedicated microservice providing health checks, model metadata, and potential synchronous preprocessing if required in the future.
- **Framework**: FastAPI (Pydantic models, async endpoints)
- **Model Registry**: Exposes current active `PRODUCTION` model parameters to the gateway.

### 3. Asynchronous Worker Pool (Python)
To prevent web request timeouts and ensure zero-data-loss processing, the ML inference is fully asynchronous.
- **Implementation**: `worker.py`
- **Responsibilities**:
  - Polls `processing_jobs` for `PENDING` states.
  - Loads the registered TensorFlow MobileNetV2 model.
  - Computes the prediction.
  - Generates Grad-CAM visual explanations (Explainable AI).
  - Updates `skin_scans` with results and transitions job state to `COMPLETED`.

### 4. Persistence Layer (MySQL 8)
Acts as the central source of truth for user state, scan lifecycle, idempotency records, and ML model versioning. See `DATABASE.md` for schema details.

## The Scan Lifecycle (Asynchronous Flow)
1. **Client**: Submits `POST /api/v1/scans` (or via web UI) with image and `Idempotency-Key`.
2. **Gateway**: Validates image, securely stores it, and inserts a `skin_scans` record with `status='QUEUED'`.
3. **Gateway**: Inserts a `processing_jobs` record with `status='PENDING'`. Returns `scan_id` to client.
4. **Worker**: Locks the job (`status='PROCESSING'`), reads the image, and executes ML inference.
5. **Worker**: Generates Grad-CAM explanation and saves the visualization.
6. **Worker**: Updates `skin_scans` (results, `status='COMPLETED'`) and unlocks job.
7. **Client**: Polls the result page or history dashboard.

## Key Engineering Decisions
- **Decoupled ML Processing**: Prevents the web server from hanging on computationally intensive tensor operations.
- **Explainability (XAI)**: Grad-CAM generates heatmaps to increase trust and transparency in the AI's decision-making process.
- **Idempotency**: Implemented via `Idempotency-Key` tracking to prevent duplicate scan creation on network retries.
- **Model Versioning**: Tracks `model_version` and `model_architecture` directly against the scan for historical accuracy.
