<?php
/**
 * DermScan AI - Scan Page (Camera + Upload)
 */

require_once __DIR__ . '/../config/config.php';

use App\Controllers\AuthController;
use App\Controllers\ScanController;

$auth = new AuthController();
$auth->requireAuth();

$scanController = new ScanController();
$userId = $auth->getCurrentUserId();

$result = null;
$error = '';

// Handle image upload
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['image'])) {
    $bodyLocation = $_POST['body_location'] ?? null;
    $notes = $_POST['notes'] ?? null;
    
    $analysis = $scanController->analyzeImage($_FILES['image'], $userId, $bodyLocation, $notes);
    
    if ($analysis['success']) {
        header('Location: result.php?scan_id=' . $analysis['scan_id']);
        exit;
    } else {
        $error = $analysis['message'];
    }
}

// Check ML service health
$mlHealthy = $scanController->checkMLServiceHealth();

// Setup Twig
$loader = new \Twig\Loader\FilesystemLoader(__DIR__ . '/../templates');
$twig = new \Twig\Environment($loader, [
    'cache' => false,
    'debug' => true
]);

echo $twig->render('scan.html.twig', [
    'user_name' => $_SESSION['user_name'] ?? 'User',
    'is_admin' => $_SESSION['is_admin'] ?? false,
    'error' => $error,
    'ml_healthy' => $mlHealthy,
    'app_name' => APP_NAME,
    'app_url' => APP_URL
]);
