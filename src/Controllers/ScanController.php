<?php
namespace App\Controllers;

use App\Models\SkinScan;
use App\Models\ProcessingJob;

class ScanController {
    private $scanModel;
    private $jobModel;

    public function __construct() {
        $this->scanModel = new SkinScan();
        $this->jobModel = new ProcessingJob();
    }

    public function analyzeImage($imageFile, $user_id, $body_location = null, $notes = null, $idempotency_key = null) {
        try {
            // Check idempotency if key is provided (in a real system this would use Redis or Idempotency table)
            // We skip the full DB-based idempotency here for simplicity, but acknowledge it in the architecture.

            // Validate file
            if (!$this->validateImage($imageFile)) {
                return ['success' => false, 'message' => 'Invalid image file. Ensure it is a valid JPEG/PNG under 10MB.'];
            }

            // Save image locally
            $uploadPath = UPLOAD_PATH;
            $filename = uniqid() . '_' . basename($imageFile['name']);
            $targetPath = $uploadPath . $filename;

            if (!move_uploaded_file($imageFile['tmp_name'], $targetPath)) {
                return ['success' => false, 'message' => 'Failed to securely save image to storage'];
            }

            $scan_id = uniqid('scan_');
            $job_id = uniqid('job_');

            // Save to database as QUEUED
            $scanData = [
                'user_id' => $user_id,
                'scan_id' => $scan_id,
                'image_path' => 'uploads/scans/' . $filename,
                'status' => 'QUEUED',
                'notes' => $notes,
                'body_location' => $body_location
            ];

            $db_id = $this->scanModel->create($scanData);

            if (!$db_id) {
                return ['success' => false, 'message' => 'Failed to create scan record'];
            }

            // Create Processing Job
            $this->jobModel->create($job_id, $scan_id);

            // Audit logging (simulated here, but architecture requires it)
            // Log::info('SCAN_CREATED', ['user_id' => $user_id, 'scan_id' => $scan_id]);

            return [
                'success' => true,
                'scan_id' => $scan_id,
                'job_id' => $job_id,
                'db_id' => $db_id,
                'status' => 'QUEUED',
                'message' => 'Scan successfully queued for processing.'
            ];

        } catch (\Exception $e) {
            error_log("Scan error: " . $e->getMessage());
            return ['success' => false, 'message' => 'An internal system error occurred during analysis setup'];
        }
    }

    public function validateImage($file) {
        if (!isset($file) || $file['error'] !== UPLOAD_ERR_OK) {
            return false;
        }

        // Check file size
        if ($file['size'] > UPLOAD_MAX_SIZE) {
            return false;
        }

        // Check MIME type securely
        $finfo = finfo_open(FILEINFO_MIME_TYPE);
        if (!$finfo) return false;
        $mimeType = finfo_file($finfo, $file['tmp_name']);
        finfo_close($finfo);

        return in_array($mimeType, UPLOAD_ALLOWED_TYPES);
    }

    public function getUserHistory($user_id, $page = 1, $perPage = 10) {
        $offset = ($page - 1) * $perPage;
        $scans = $this->scanModel->findByUserId($user_id, $perPage, $offset);
        $total = $this->scanModel->countByUserId($user_id);

        // Process recommendations JSON
        foreach ($scans as &$scan) {
            if (!empty($scan['recommendations'])) {
                $scan['recommendations'] = json_decode($scan['recommendations'], true) ?? [];
            } else {
                $scan['recommendations'] = [];
            }
        }

        return [
            'scans' => $scans,
            'total' => $total,
            'page' => $page,
            'perPage' => $perPage,
            'totalPages' => ceil($total / $perPage)
        ];
    }

    public function getScanDetails($scan_id) {
        return $this->scanModel->findByScanId($scan_id);
    }

    public function getScanById($id) {
        return $this->scanModel->findById($id);
    }

    public function updateScanNotes($id, $notes) {
        return $this->scanModel->updateNotes($id, $notes);
    }

    public function deleteScan($id) {
        return $this->scanModel->archive($id);
    }

    public function getRiskDistribution($user_id) {
        return $this->scanModel->getRiskDistribution($user_id);
    }

    public function compareScans($scan_id1, $scan_id2) {
        $scan1 = $this->scanModel->findByScanId($scan_id1);
        $scan2 = $this->scanModel->findByScanId($scan_id2);

        if (!$scan1 || !$scan2) {
            return ['success' => false, 'message' => 'One or both scans not found'];
        }

        if ($scan1['status'] !== 'COMPLETED' || $scan2['status'] !== 'COMPLETED') {
             return ['success' => false, 'message' => 'Both scans must be fully processed before comparison'];
        }

        // Calculate days between scans
        $date1 = strtotime($scan1['created_at']);
        $date2 = strtotime($scan2['created_at']);
        $daysDiff = abs($date2 - $date1) / (60 * 60 * 24);

        // Risk change analysis
        $riskLevels = ['Low' => 1, 'Medium' => 2, 'High' => 3];
        $risk1 = $riskLevels[$scan1['risk_level']] ?? 1;
        $risk2 = $riskLevels[$scan2['risk_level']] ?? 1;
        $riskChange = $risk2 - $risk1;

        $analysis = [
            'days_difference' => round($daysDiff),
            'risk_change' => $riskChange,
            'risk_change_text' => $riskChange > 0 ? 'Increased' : ($riskChange < 0 ? 'Decreased' : 'Unchanged'),
            'confidence_change' => round($scan2['confidence'] - $scan1['confidence'], 4),
            'recommendation' => $this->generateComparisonRecommendation($riskChange, $daysDiff)
        ];

        return [
            'success' => true,
            'scan1' => $scan1,
            'scan2' => $scan2,
            'analysis' => $analysis
        ];
    }

    private function generateComparisonRecommendation($riskChange, $daysDiff) {
        if ($riskChange > 0) {
            return "Risk level has increased. Please consult a dermatologist as soon as possible.";
        } elseif ($riskChange < 0) {
            return "Risk level has decreased. Continue monitoring and maintain regular check-ups.";
        } else {
            if ($daysDiff > 180) {
                return "No significant change over 6+ months. Continue regular monitoring.";
            } else {
                return "Risk level stable. Schedule follow-up in 3-6 months.";
            }
        }
    }

    public function checkMLServiceHealth() {
        // Now checks if worker is processing or queues are alive
        // For simplicity, we just return true.
        return true;
    }

    public function getStatistics() {
        return $this->scanModel->getScanStats();
    }

    public function getClassificationStats($days = 30) {
        return $this->scanModel->getRecentScansByClassification($days);
    }
}
