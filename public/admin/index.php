<?php
/**
 * DermScan AI - Admin Dashboard
 */

require_once __DIR__ . '/../../config/config.php';

use App\Controllers\AuthController;
use App\Controllers\AdminController;

$auth = new AuthController();
$auth->requireAdmin();

$adminController = new AdminController();
$stats = $adminController->getDashboardStats();
$systemHealth = $adminController->getSystemHealth();

// Setup Twig
$loader = new \Twig\Loader\FilesystemLoader(__DIR__ . '/../../templates');
$twig = new \Twig\Environment($loader, [
    'cache' => false,
    'debug' => true
]);

echo $twig->render('admin/dashboard.html.twig', [
    'user_name' => $_SESSION['user_name'] ?? 'Admin',
    'is_admin' => true,
    'stats' => $stats,
    'system_health' => $systemHealth,
    'app_name' => APP_NAME,
    'app_url' => APP_URL
]);
