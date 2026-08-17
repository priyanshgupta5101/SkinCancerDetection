<?php
/**
 * DermScan AI - Main Entry Point
 */

require_once __DIR__ . '/../config/config.php';

use App\Controllers\AuthController;

$auth = new AuthController();

// Redirect to dashboard if logged in, otherwise to login
if ($auth->isLoggedIn()) {
    header('Location: dashboard.php');
} else {
    header('Location: login.php');
}
exit;
