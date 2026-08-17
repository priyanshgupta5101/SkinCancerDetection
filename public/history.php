<?php
/**
 * DermScan AI - Scan History Page
 */

require_once __DIR__ . '/../config/config.php';

use App\Controllers\AuthController;
use App\Controllers\ScanController;

$auth = new AuthController();
$auth->requireAuth();

$scanController = new ScanController();
$userId = $auth->getCurrentUserId();

$page = intval($_GET['page'] ?? 1);
$history = $scanController->getUserHistory($userId, $page, 10);

// Handle delete action
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['action']) && $_POST['action'] === 'delete') {
    $scanId = intval($_POST['scan_id'] ?? 0);
    if ($scanId) {
        $scanController->deleteScan($scanId);
        header('Location: history.php?page=' . $page);
        exit;
    }
}

// Setup Twig
$loader = new \Twig\Loader\FilesystemLoader(__DIR__ . '/../templates');
$twig = new \Twig\Environment($loader, [
    'cache' => false,
    'debug' => true
]);

echo $twig->render('history.html.twig', [
    'user_name' => $_SESSION['user_name'] ?? 'User',
    'is_admin' => $_SESSION['is_admin'] ?? false,
    'scans' => $history['scans'],
    'total' => $history['total'],
    'page' => $history['page'],
    'perPage' => $history['perPage'],
    'totalPages' => $history['totalPages'],
    'app_name' => APP_NAME,
    'app_url' => APP_URL
]);
