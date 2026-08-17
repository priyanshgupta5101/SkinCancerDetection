# Database Schema Documentation

## Philosophy
The DermScan AI relational database follows strict normalization where possible while leveraging JSON columns for flexible metadata (e.g., recommendations, system configurations). The schema supports distributed job processing, model versioning, and comprehensive system auditing.

## Core Tables

### `users`
Core user entity managing identity, demographics, and clinical history (e.g., `skin_type`, `family_history`).

### `models` (Model Registry)
Tracks the lifecycle of ML models deployed to the platform.
- **Fields**: `model_version`, `model_architecture`, `model_checksum`, `preprocessing_version`, `status` (`CANDIDATE`, `PRODUCTION`, etc.)
- **Purpose**: Ensures deterministic historical tracking. If a scan was processed on `v1.0.0`, it remains tied to that version for clinical review.

### `skin_scans`
The primary domain entity for user uploaded skin lesions.
- **Fields**: `scan_id`, `status` (`QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`), `classification`, `risk_level`, `confidence`, `explanation_path` (Grad-CAM), `model_version`.
- **Purpose**: Holds the asynchronous state and the final clinical assessment.

### `processing_jobs`
Handles the asynchronous job queue.
- **Fields**: `job_id`, `scan_id`, `status` (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`), `worker_id`, `retry_count`.
- **Purpose**: Allows horizontal scaling of workers. Workers lock rows by updating `worker_id` and setting `status='PROCESSING'`.

### `idempotency_records`
Ensures safe retries from the client.
- **Fields**: `idempotency_key`, `path`, `response_code`, `response_body`.
- **Purpose**: If a client POSTs the same scan multiple times due to a dropped connection, the gateway returns the cached `response_body`.

### `audit_events`
Provides non-repudiation and security tracking.
- **Fields**: `actor_id`, `actor_type`, `action` (e.g., `SCAN_COMPLETED`, `MODEL_UPDATED`), `resource_type`, `resource_id`.

## Security Considerations
- **Prepared Statements**: All database queries utilize PDO parameter binding.
- **Password Hashing**: BCrypt (`PASSWORD_DEFAULT`) is enforced.
- **Data Minimization**: The application explicitly tracks `privacy_opt_in` to handle data retention policies correctly.
