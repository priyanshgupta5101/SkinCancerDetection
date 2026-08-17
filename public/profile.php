<?php
/**
 * DermScan AI - User Profile Page
 */

require_once __DIR__ . '/../config/config.php';

use App\Controllers\AuthController;

$auth = new AuthController();
$auth->requireAuth();

$userId = $auth->getCurrentUserId();
$profile = $auth->getProfile($userId);

$success = '';
$error = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $data = [
        'first_name' => $_POST['first_name'] ?? $profile['first_name'],
        'last_name' => $_POST['last_name'] ?? $profile['last_name'],
        'age' => $_POST['age'] ?? $profile['age'],
        'skin_type' => $_POST['skin_type'] ?? $profile['skin_type'],
        'family_history' => isset($_POST['family_history']) ? true : false,
        'privacy_opt_in' => isset($_POST['privacy_opt_in']) ? true : false
    ];
    
    if ($auth->updateProfile($userId, $data)) {
        $success = 'Profile updated successfully';
        $profile = $auth->getProfile($userId);
        $_SESSION['user_name'] = $profile['first_name'] . ' ' . $profile['last_name'];
    } else {
        $error = 'Failed to update profile';
    }
}

// Setup Twig
$loader = new \Twig\Loader\FilesystemLoader(__DIR__ . '/../templates');
$twig = new \Twig\Environment($loader, [
    'cache' => false,
    'debug' => true
]);

echo $twig->render('profile.html.twig', [
    'user_name' => $_SESSION['user_name'] ?? 'User',
    'is_admin' => $_SESSION['is_admin'] ?? false,
    'profile' => $profile,
    'success' => $success,
    'error' => $error,
    'app_name' => APP_NAME,
    'app_url' => APP_URL
]);
