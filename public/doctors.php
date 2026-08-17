<?php
/**
 * DermScan AI - Find Doctors Page
 */

require_once __DIR__ . '/../config/config.php';

use App\Controllers\AuthController;
use App\Controllers\AdminController;

$auth = new AuthController();
$auth->requireAuth();

$adminController = new AdminController();

$selectedCity = $_GET['city'] ?? null;
$cities = $adminController->getDoctorCities();
$doctors = $adminController->getDoctors($selectedCity);

// Setup Twig
$loader = new \Twig\Loader\FilesystemLoader(__DIR__ . '/../templates');
$twig = new \Twig\Environment($loader, [
    'cache' => false,
    'debug' => true
]);

echo $twig->render('doctors.html.twig', [
    'user_name' => $_SESSION['user_name'] ?? 'User',
    'is_admin' => $_SESSION['is_admin'] ?? false,
    'doctors' => $doctors,
    'cities' => $cities,
    'selected_city' => $selectedCity,
    'app_name' => APP_NAME,
    'app_url' => APP_URL
]);
