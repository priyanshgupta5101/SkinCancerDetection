<?php
namespace App\Controllers;

use App\Models\User;

class AuthController {
    private $userModel;

    public function __construct() {
        $this->userModel = new User();
    }

    public function login($email, $password) {
        $ip_address = $_SERVER['REMOTE_ADDR'] ?? 'unknown';
        
        // Check for too many failed attempts
        $recentFailures = $this->userModel->getRecentFailedAttempts($email, $ip_address);
        if ($recentFailures >= MAX_LOGIN_ATTEMPTS) {
            return ['success' => false, 'message' => 'Too many failed attempts. Please try again later.'];
        }

        $user = $this->userModel->findByEmail($email);
        
        if (!$user) {
            $this->userModel->recordLoginAttempt($email, $ip_address, false);
            return ['success' => false, 'message' => 'Invalid credentials'];
        }

        if (!$user['is_active']) {
            return ['success' => false, 'message' => 'Account is deactivated'];
        }

        if (!$this->userModel->verifyPassword($password, $user['password_hash'])) {
            $this->userModel->recordLoginAttempt($email, $ip_address, false);
            return ['success' => false, 'message' => 'Invalid credentials'];
        }

        // Successful login
        $this->userModel->recordLoginAttempt($email, $ip_address, true);
        $this->userModel->updateLastLogin($user['id']);

        // Set session
        $_SESSION['user_id'] = $user['id'];
        $_SESSION['user_email'] = $user['email'];
        $_SESSION['user_name'] = $user['first_name'] . ' ' . $user['last_name'];
        $_SESSION['is_admin'] = (bool)$user['is_admin'];
        $_SESSION['logged_in'] = true;
        $_SESSION['login_time'] = time();

        return [
            'success' => true, 
            'user' => [
                'id' => $user['id'],
                'email' => $user['email'],
                'name' => $user['first_name'] . ' ' . $user['last_name'],
                'is_admin' => (bool)$user['is_admin']
            ]
        ];
    }

    public function register($data) {
        // Validate required fields
        $required = ['email', 'password', 'first_name', 'last_name'];
        foreach ($required as $field) {
            if (empty($data[$field])) {
                return ['success' => false, 'message' => "Field '$field' is required"];
            }
        }

        // Validate email
        if (!filter_var($data['email'], FILTER_VALIDATE_EMAIL)) {
            return ['success' => false, 'message' => 'Invalid email format'];
        }

        // Check if email exists
        $existing = $this->userModel->findByEmail($data['email']);
        if ($existing) {
            return ['success' => false, 'message' => 'Email already registered'];
        }

        // Password validation
        if (strlen($data['password']) < 8) {
            return ['success' => false, 'message' => 'Password must be at least 8 characters'];
        }

        // Create user
        $userId = $this->userModel->create($data);
        
        if ($userId) {
            return ['success' => true, 'user_id' => $userId, 'message' => 'Registration successful'];
        }

        return ['success' => false, 'message' => 'Registration failed'];
    }

    public function logout() {
        // Clear session
        $_SESSION = [];
        session_destroy();
        return ['success' => true, 'message' => 'Logged out successfully'];
    }

    public function isLoggedIn() {
        return isset($_SESSION['logged_in']) && $_SESSION['logged_in'] === true;
    }

    public function isAdmin() {
        return $this->isLoggedIn() && isset($_SESSION['is_admin']) && $_SESSION['is_admin'] === true;
    }

    public function getCurrentUserId() {
        return $_SESSION['user_id'] ?? null;
    }

    public function requireAuth() {
        if (!$this->isLoggedIn()) {
            header('Location: ' . APP_URL . '/login.php');
            exit;
        }

        // Check session expiration
        if (isset($_SESSION['login_time']) && (time() - $_SESSION['login_time'] > SESSION_LIFETIME)) {
            $this->logout();
            header('Location: ' . APP_URL . '/login.php?expired=1');
            exit;
        }

        // Update session time
        $_SESSION['login_time'] = time();
    }

    public function requireAdmin() {
        $this->requireAuth();
        if (!$this->isAdmin()) {
            header('Location: ' . APP_URL . '/dashboard.php');
            exit;
        }
    }

    public function updateProfile($user_id, $data) {
        return $this->userModel->update($user_id, $data);
    }

    public function getProfile($user_id) {
        return $this->userModel->findById($user_id);
    }
}
