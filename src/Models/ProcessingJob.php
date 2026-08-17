<?php
namespace App\Models;

require_once __DIR__ . '/../../config/database.php';

class ProcessingJob {
    private $db;
    private $table = 'processing_jobs';

    public function __construct() {
        $database = new \Database();
        $this->db = $database->connect();
    }

    public function create($job_id, $scan_id) {
        $query = "INSERT INTO " . $this->table . " (job_id, scan_id, status) VALUES (:job_id, :scan_id, 'PENDING')";
        $stmt = $this->db->prepare($query);
        $stmt->bindValue(':job_id', $job_id);
        $stmt->bindValue(':scan_id', $scan_id);
        return $stmt->execute();
    }

    public function updateStatus($job_id, $status, $error_message = null) {
        $query = "UPDATE " . $this->table . " SET status = :status, error_message = :error_message";
        
        if ($status === 'PROCESSING') {
            $query .= ", started_at = CURRENT_TIMESTAMP";
        } elseif (in_array($status, ['COMPLETED', 'FAILED', 'ABANDONED'])) {
            $query .= ", completed_at = CURRENT_TIMESTAMP";
        }
        
        $query .= " WHERE job_id = :job_id";
        
        $stmt = $this->db->prepare($query);
        $stmt->bindValue(':status', $status);
        $stmt->bindValue(':error_message', $error_message);
        $stmt->bindValue(':job_id', $job_id);
        return $stmt->execute();
    }

    public function findByJobId($job_id) {
        $query = "SELECT * FROM " . $this->table . " WHERE job_id = :job_id LIMIT 1";
        $stmt = $this->db->prepare($query);
        $stmt->bindValue(':job_id', $job_id);
        $stmt->execute();
        return $stmt->fetch(\PDO::FETCH_ASSOC);
    }
}
