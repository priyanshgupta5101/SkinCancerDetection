<?php
/**
 * DermScan AI - Application Configuration
 */

// Start session
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

// Error reporting
error_reporting(E_ALL);
ini_set('display_errors', '1');

// Timezone
date_default_timezone_set('UTC');

// Application settings
define('APP_NAME', 'DermScan AI');
define('APP_VERSION', '1.0.0');
define('APP_URL', 'http://localhost/DermScan%20AI/public');

// ML Service Configuration
define('ML_SERVICE_URL', $_ENV['ML_SERVICE_URL'] ?? 'http://127.0.0.1:8000');
define('ML_API_TIMEOUT', 30);

// File Upload Configuration
define('UPLOAD_MAX_SIZE', 10 * 1024 * 1024); // 10MB
define('UPLOAD_ALLOWED_TYPES', ['image/jpeg', 'image/png', 'image/jpg']);
define('UPLOAD_PATH', __DIR__ . '/../public/uploads/scans/');

// Security
define('SESSION_LIFETIME', 3600); // 1 hour
define('MAX_LOGIN_ATTEMPTS', 5);
define('LOCKOUT_TIME', 900); // 15 minutes

// Create upload directory if not exists
if (!is_dir(UPLOAD_PATH)) {
    mkdir(UPLOAD_PATH, 0755, true);
}

// Autoloader
require_once __DIR__ . '/../vendor/autoload.php';
