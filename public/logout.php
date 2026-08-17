<?php
/**
 * DermScan AI - Logout
 */

require_once __DIR__ . '/../config/config.php';

use App\Controllers\AuthController;

$auth = new AuthController();
$auth->logout();

header('Location: login.php');
exit;
