<?php
/**
 * DermScan AI - User Dashboard
 */

require_once __DIR__ . '/../config/config.php';

use App\Controllers\AuthController;
use App\Controllers\ScanController;

$auth = new AuthController();
$auth->requireAuth();

$scanController = new ScanController();
$userId = $auth->getCurrentUserId();

// Get user scan history
$history = $scanController->getUserHistory($userId, 1, 5);
$riskDistribution = $scanController->getRiskDistribution($userId);

// Setup Twig
$loader = new \Twig\Loader\FilesystemLoader(__DIR__ . '/../templates');
$twig = new \Twig\Environment($loader, [
    'cache' => false,
    'debug' => true
]);

echo $twig->render('dashboard.html.twig', [
    'user_name' => $_SESSION['user_name'] ?? 'User',
    'is_admin' => $_SESSION['is_admin'] ?? false,
    'scans' => $history['scans'],
    'total_scans' => $history['total'],
    'risk_distribution' => $riskDistribution,
    'app_name' => APP_NAME,
    'app_url' => APP_URL
]);
