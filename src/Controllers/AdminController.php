<?php
namespace App\Controllers;

use App\Models\User;
use App\Models\SkinScan;
use App\Models\Doctor;

class AdminController {
    private $userModel;
    private $scanModel;
    private $doctorModel;

    public function __construct() {
        $this->userModel = new User();
        $this->scanModel = new SkinScan();
        $this->doctorModel = new Doctor();
    }

    public function getDashboardStats() {
        $userStats = [
            'total_users' => $this->userModel->countAllUsers(),
            'new_users_today' => $this->getNewUsersToday(),
            'active_users' => $this->getActiveUsersLast7Days()
        ];

        $scanStats = $this->scanModel->getScanStats();
        
        $riskDistribution = [
            'labels' => ['High', 'Medium', 'Low'],
            'data' => [
                $scanStats['high_risk'] ?? 0,
                $scanStats['medium_risk'] ?? 0,
                $scanStats['low_risk'] ?? 0
            ]
        ];

        $recentClassifications = $this->scanModel->getRecentScansByClassification(7);

        return [
            'users' => $userStats,
            'scans' => $scanStats,
            'risk_distribution' => $riskDistribution,
            'recent_classifications' => $recentClassifications
        ];
    }

    private function getNewUsersToday() {
        // This would query users created today
        return 0; // Placeholder
    }

    private function getActiveUsersLast7Days() {
        // This would query users with login in last 7 days
        return 0; // Placeholder
    }

    public function getAllUsers($page = 1, $perPage = 20) {
        $offset = ($page - 1) * $perPage;
        $users = $this->userModel->getAllUsers($perPage, $offset);
        $total = $this->userModel->countAllUsers();

        return [
            'users' => $users,
            'total' => $total,
            'page' => $page,
            'perPage' => $perPage,
            'totalPages' => ceil($total / $perPage)
        ];
    }

    public function getAllScans($page = 1, $perPage = 50) {
        $offset = ($page - 1) * $perPage;
        $scans = $this->scanModel->getAllScans($perPage, $offset);
        $stats = $this->scanModel->getScanStats();

        return [
            'scans' => $scans,
            'total' => $stats['total_scans'] ?? 0,
            'page' => $page,
            'perPage' => $perPage,
            'totalPages' => ceil(($stats['total_scans'] ?? 0) / $perPage)
        ];
    }

    public function getDoctors($city = null) {
        return $this->doctorModel->findAll($city);
    }

    public function addDoctor($data) {
        return $this->doctorModel->create($data);
    }

    public function updateDoctor($id, $data) {
        return $this->doctorModel->update($id, $data);
    }

    public function deleteDoctor($id) {
        return $this->doctorModel->delete($id);
    }

    public function getDoctorCities() {
        return $this->doctorModel->getCities();
    }

    public function logAdminAction($admin_id, $action, $target_type = null, $target_id = null, $details = null) {
        // Implementation would insert into admin_logs table
        return true;
    }

    public function archiveOldScans($days = 365) {
        // Implementation would archive scans older than specified days
        return ['archived' => 0];
    }

    public function getSystemHealth() {
        // Check database connection
        $dbHealth = true;
        try {
            new \Database();
        } catch (\Exception $e) {
            $dbHealth = false;
        }

        // Check ML service
        $scanController = new ScanController();
        $mlHealth = $scanController->checkMLServiceHealth();

        return [
            'database' => $dbHealth ? 'healthy' : 'error',
            'ml_service' => $mlHealth ? 'healthy' : 'error',
            'storage' => $this->checkStorageHealth(),
            'timestamp' => date('Y-m-d H:i:s')
        ];
    }

    private function checkStorageHealth() {
        $uploadDir = UPLOAD_PATH;
        if (!is_dir($uploadDir)) {
            return 'error';
        }
        
        $freeSpace = disk_free_space($uploadDir);
        $totalSpace = disk_total_space($uploadDir);
        $usagePercent = (($totalSpace - $freeSpace) / $totalSpace) * 100;
        
        if ($usagePercent > 90) {
            return 'critical';
        } elseif ($usagePercent > 75) {
            return 'warning';
        }
        return 'healthy';
    }
}
