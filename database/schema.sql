-- DermScan AI - Database Schema
-- MySQL Database Structure

-- Create database
CREATE DATABASE IF NOT EXISTS dermscan_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE dermscan_db;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    age INT,
    skin_type ENUM('fair', 'medium', 'olive', 'brown', 'dark') DEFAULT 'medium',
    family_history BOOLEAN DEFAULT FALSE,
    privacy_opt_in BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    email_verified_at TIMESTAMP NULL,
    last_login_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_active (is_active)
) ENGINE=InnoDB;

-- ML Models Registry table
CREATE TABLE IF NOT EXISTS models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL UNIQUE,
    model_architecture VARCHAR(100) NOT NULL,
    model_checksum VARCHAR(255) NOT NULL,
    preprocessing_version VARCHAR(50),
    status ENUM('CANDIDATE', 'VALIDATING', 'APPROVED', 'STAGING', 'PRODUCTION', 'DEPRECATED', 'RETIRED') DEFAULT 'CANDIDATE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_model_version (model_version)
) ENGINE=InnoDB;

-- Skin scans table
CREATE TABLE IF NOT EXISTS skin_scans (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    scan_id VARCHAR(255) NOT NULL UNIQUE,
    image_path VARCHAR(500) NOT NULL,
    status ENUM('UPLOADED', 'VALIDATING', 'VALIDATED', 'QUEUED', 'PROCESSING', 'EXPLAINING', 'COMPLETED', 'FAILED') DEFAULT 'UPLOADED',
    classification VARCHAR(100),
    risk_level ENUM('Low', 'Medium', 'High'),
    confidence DECIMAL(5,4),
    description TEXT,
    recommendations JSON,
    notes TEXT,
    body_location VARCHAR(100),
    compared_to_scan_id VARCHAR(255),
    explanation_path VARCHAR(500),
    model_version VARCHAR(50),
    error_message TEXT,
    is_archived BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (model_version) REFERENCES models(model_version) ON DELETE SET NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_scan_id (scan_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- Processing Jobs table
CREATE TABLE IF NOT EXISTS processing_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    job_id VARCHAR(255) NOT NULL UNIQUE,
    scan_id VARCHAR(255) NOT NULL,
    status ENUM('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'RETRYING', 'ABANDONED') DEFAULT 'PENDING',
    retry_count INT DEFAULT 0,
    worker_id VARCHAR(100),
    error_message TEXT,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES skin_scans(scan_id) ON DELETE CASCADE,
    INDEX idx_scan_id (scan_id),
    INDEX idx_status (status),
    INDEX idx_worker_id (worker_id)
) ENGINE=InnoDB;

-- Idempotency Records table
CREATE TABLE IF NOT EXISTS idempotency_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    idempotency_key VARCHAR(255) NOT NULL UNIQUE,
    path VARCHAR(255) NOT NULL,
    response_code INT,
    response_body JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_idempotency_key (idempotency_key),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- Audit Logs table
CREATE TABLE IF NOT EXISTS audit_events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    actor_id INT,
    actor_type VARCHAR(50),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    details JSON,
    ip_address VARCHAR(45),
    request_id VARCHAR(255),
    result ENUM('SUCCESS', 'FAILURE') DEFAULT 'SUCCESS',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_actor_id (actor_id),
    INDEX idx_action (action),
    INDEX idx_created_at (created_at),
    INDEX idx_request_id (request_id)
) ENGINE=InnoDB;

-- Login attempts table (security)
CREATE TABLE IF NOT EXISTS login_attempts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    ip_address VARCHAR(45) NOT NULL,
    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_successful BOOLEAN DEFAULT FALSE,
    INDEX idx_email (email),
    INDEX idx_ip (ip_address),
    INDEX idx_attempted (attempted_at)
) ENGINE=InnoDB;

-- System settings
CREATE TABLE IF NOT EXISTS system_settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(100) NOT NULL UNIQUE,
    setting_value TEXT,
    setting_type ENUM('string', 'int', 'bool', 'json') DEFAULT 'string',
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    updated_by INT,
    INDEX idx_setting_key (setting_key)
) ENGINE=InnoDB;

-- Password reset tokens
CREATE TABLE IF NOT EXISTS password_resets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    token VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    used_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_token (token),
    INDEX idx_expires (expires_at)
) ENGINE=InnoDB;

-- Doctor directory
CREATE TABLE IF NOT EXISTS doctors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    specialty VARCHAR(100) NOT NULL,
    clinic_name VARCHAR(200),
    address TEXT,
    city VARCHAR(100),
    phone VARCHAR(50),
    email VARCHAR(255),
    website VARCHAR(255),
    is_verified BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_city (city),
    INDEX idx_specialty (specialty),
    INDEX idx_active (is_active)
) ENGINE=InnoDB;

-- ML Model performance tracking
CREATE TABLE IF NOT EXISTS ml_model_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    scan_id VARCHAR(255),
    processing_time_ms INT,
    confidence_score DECIMAL(5,4),
    was_correct BOOLEAN,
    feedback_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_version) REFERENCES models(model_version) ON DELETE CASCADE,
    INDEX idx_model_version (model_version),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- Insert initial model version
INSERT INTO models (model_version, model_architecture, model_checksum, preprocessing_version, status)
VALUES ('v1.0.0', 'MobileNetV2', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'v1', 'PRODUCTION')
ON DUPLICATE KEY UPDATE id=id;

-- Insert default admin user (password: admin123 - change in production!)
INSERT INTO users (email, password_hash, first_name, last_name, is_admin, email_verified_at) 
VALUES (
    'admin@dermscan.ai', 
    '$2y$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', -- admin123
    'System', 
    'Administrator', 
    TRUE, 
    NOW()
) ON DUPLICATE KEY UPDATE id=id;

-- Insert default system settings
INSERT INTO system_settings (setting_key, setting_value, setting_type, description) VALUES
('site_name', 'DermScan AI', 'string', 'Application name displayed in UI'),
('allow_registration', 'true', 'bool', 'Allow new user registrations'),
('maintenance_mode', 'false', 'bool', 'Enable maintenance mode'),
('ml_service_enabled', 'true', 'bool', 'Enable ML analysis service'),
('max_upload_size_mb', '10', 'int', 'Maximum upload size in MB'),
('retention_days', '365', 'int', 'Days to retain scan data'),
('default_theme', 'light', 'string', 'Default UI theme (light/dark)')
ON DUPLICATE KEY UPDATE id=id;

-- Insert sample doctors
INSERT INTO doctors (name, specialty, clinic_name, city, phone, email, is_verified) VALUES
('Dr. Sarah Johnson', 'Dermatology', 'City Skin Care Center', 'New York', '+1-555-0101', 'dr.johnson@cityskin.example', TRUE),
('Dr. Michael Chen', 'Dermatology', 'Advanced Dermatology Institute', 'Los Angeles', '+1-555-0102', 'dr.chen@adiderm.example', TRUE),
('Dr. Emily Rodriguez', 'Dermatology', 'Skin Health Partners', 'Chicago', '+1-555-0103', 'dr.rodriguez@skinp.example', TRUE),
('Dr. James Wilson', 'Dermatology', 'Metro Dermatology Clinic', 'Houston', '+1-555-0104', 'dr.wilson@metrod.example', TRUE),
('Dr. Lisa Park', 'Dermatology', 'Seattle Skin Specialists', 'Seattle', '+1-555-0105', 'dr.park@seattleskin.example', TRUE)
ON DUPLICATE KEY UPDATE id=id;

