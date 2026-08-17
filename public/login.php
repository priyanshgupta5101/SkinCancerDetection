<?php
/**
 * DermScan AI - Login Page
 */

require_once __DIR__ . '/../config/config.php';

use App\Controllers\AuthController;

$auth = new AuthController();

// If already logged in, redirect to dashboard
if ($auth->isLoggedIn()) {
    header('Location: dashboard.php');
    exit;
}

$error = '';
$success = '';

// Handle login form
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['action']) && $_POST['action'] === 'login') {
    $result = $auth->login($_POST['email'] ?? '', $_POST['password'] ?? '');
    
    if ($result['success']) {
        header('Location: dashboard.php');
        exit;
    } else {
        $error = $result['message'];
    }
}

// Handle registration form
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['action']) && $_POST['action'] === 'register') {
    $data = [
        'email' => $_POST['email'] ?? '',
        'password' => $_POST['password'] ?? '',
        'first_name' => $_POST['first_name'] ?? '',
        'last_name' => $_POST['last_name'] ?? '',
        'age' => $_POST['age'] ?? null,
        'skin_type' => $_POST['skin_type'] ?? 'medium',
        'family_history' => isset($_POST['family_history']) ? true : false,
        'privacy_opt_in' => isset($_POST['privacy_opt_in']) ? true : false
    ];
    
    $result = $auth->register($data);
    
    if ($result['success']) {
        $success = 'Registration successful! Please login.';
    } else {
        $error = $result['message'];
        $old_data = $data;
        $active_tab = 'register';
    }
}

// Check for session expired
if (isset($_GET['expired']) && $_GET['expired'] == '1') {
    $error = 'Your session has expired. Please login again.';
}

// Setup Twig
$loader = new \Twig\Loader\FilesystemLoader(__DIR__ . '/../templates');
$twig = new \Twig\Environment($loader, [
    'cache' => false,
    'debug' => true
]);

echo $twig->render('login.html.twig', [
    'error' => $error,
    'success' => $success,
    'app_name' => APP_NAME,
    'old' => $old_data ?? [],
    'active_tab' => $active_tab ?? 'login'
]);
