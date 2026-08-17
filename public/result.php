<?php
/**
 * DermScan AI - Scan Result Page
 */

require_once __DIR__ . '/../config/config.php';

use App\Controllers\AuthController;
use App\Controllers\ScanController;

$auth = new AuthController();
$auth->requireAuth();

$scanController = new ScanController();

$scanId = $_GET['scan_id'] ?? null;
$scan = null;
$error = '';

if ($scanId) {
    $scan = $scanController->getScanDetails($scanId);
    if (!$scan) {
        $error = 'Scan not found';
    }
} else {
    $error = 'No scan ID provided';
}

// Setup Twig
$loader = new \Twig\Loader\FilesystemLoader(__DIR__ . '/../templates');
$twig = new \Twig\Environment($loader, [
    'cache' => false,
    'debug' => true
]);

echo $twig->render('result.html.twig', [
    'user_name' => $_SESSION['user_name'] ?? 'User',
    'is_admin' => $_SESSION['is_admin'] ?? false,
    'scan' => $scan,
    'error' => $error,
    'app_name' => APP_NAME,
    'app_url' => APP_URL
]);
