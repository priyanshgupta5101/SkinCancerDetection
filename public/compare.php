<?php
/**
 * DermScan AI - Compare Scans Page
 */

require_once __DIR__ . '/../config/config.php';

use App\Controllers\AuthController;
use App\Controllers\ScanController;

$auth = new AuthController();
$auth->requireAuth();

$scanController = new ScanController();
$userId = $auth->getCurrentUserId();

$scan1Id = $_GET['scan1'] ?? null;
$scan2Id = $_GET['scan2'] ?? null;
$comparison = null;
$error = '';

// Get user's scans for selection
$history = $scanController->getUserHistory($userId, 1, 100);

if ($scan1Id && $scan2Id) {
    $comparison = $scanController->compareScans($scan1Id, $scan2Id);
    if (!$comparison['success']) {
        $error = $comparison['message'];
    }
}

// Setup Twig
$loader = new \Twig\Loader\FilesystemLoader(__DIR__ . '/../templates');
$twig = new \Twig\Environment($loader, [
    'cache' => false,
    'debug' => true
]);

echo $twig->render('compare.html.twig', [
    'user_name' => $_SESSION['user_name'] ?? 'User',
    'is_admin' => $_SESSION['is_admin'] ?? false,
    'scans' => $history['scans'],
    'comparison' => $comparison,
    'selected_scan1' => $scan1Id,
    'selected_scan2' => $scan2Id,
    'error' => $error,
    'app_name' => APP_NAME,
    'app_url' => APP_URL
]);
