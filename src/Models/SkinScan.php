<?php
namespace App\Models;

require_once __DIR__ . '/../../config/database.php';

class SkinScan {
    private $db;
    private $table = 'skin_scans';

    public function __construct() {
        $database = new \Database();
        $this->db = $database->connect();
    }

    public function create($data) {
        $query = "INSERT INTO " . $this->table . " 
                  (user_id, scan_id, image_path, status, notes, body_location) 
                  VALUES 
                  (:user_id, :scan_id, :image_path, :status, :notes, :body_location)";

        $stmt = $this->db->prepare($query);
        
        $stmt->bindValue(':user_id', $data['user_id'], \PDO::PARAM_INT);
        $stmt->bindValue(':scan_id', $data['scan_id']);
        $stmt->bindValue(':image_path', $data['image_path']);
        $stmt->bindValue(':status', $data['status'] ?? 'UPLOADED');
        $stmt->bindValue(':notes', $data['notes']);
        $stmt->bindValue(':body_location', $data['body_location']);

        return $stmt->execute() ? $this->db->lastInsertId() : false;
    }

    public function updateResult($scan_id, $data) {
        $query = "UPDATE " . $this->table . " 
                  SET classification = :classification, 
                      risk_level = :risk_level, 
                      confidence = :confidence, 
                      description = :description, 
                      recommendations = :recommendations, 
                      status = :status,
                      explanation_path = :explanation_path,
                      model_version = :model_version,
                      error_message = :error_message
                  WHERE scan_id = :scan_id";

        $stmt = $this->db->prepare($query);
        
        $recommendations_json = json_encode($data['recommendations'] ?? []);

        $stmt->bindValue(':classification', $data['classification']);
        $stmt->bindValue(':risk_level', $data['risk_level']);
        $stmt->bindValue(':confidence', $data['confidence']);
        $stmt->bindValue(':description', $data['description']);
        $stmt->bindValue(':recommendations', $recommendations_json);
        $stmt->bindValue(':status', $data['status']);
        $stmt->bindValue(':explanation_path', $data['explanation_path'] ?? null);
        $stmt->bindValue(':model_version', $data['model_version'] ?? null);
        $stmt->bindValue(':error_message', $data['error_message'] ?? null);
        $stmt->bindValue(':scan_id', $scan_id);

        return $stmt->execute();
    }

    public function findByUserId($user_id, $limit = 50, $offset = 0) {
        $query = "SELECT * FROM " . $this->table . " 
                  WHERE user_id = :user_id AND is_archived = FALSE 
                  ORDER BY created_at DESC 
                  LIMIT :limit OFFSET :offset";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':user_id', $user_id, \PDO::PARAM_INT);
        $stmt->bindParam(':limit', $limit, \PDO::PARAM_INT);
        $stmt->bindParam(':offset', $offset, \PDO::PARAM_INT);
        $stmt->execute();
        return $stmt->fetchAll(\PDO::FETCH_ASSOC);
    }

    public function findByScanId($scan_id) {
        $query = "SELECT * FROM " . $this->table . " WHERE scan_id = :scan_id LIMIT 1";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':scan_id', $scan_id);
        $stmt->execute();
        $result = $stmt->fetch(\PDO::FETCH_ASSOC);
        if ($result && !empty($result['recommendations'])) {
            $result['recommendations'] = json_decode($result['recommendations'], true) ?? [];
        }
        return $result;
    }

    public function findById($id) {
        $query = "SELECT * FROM " . $this->table . " WHERE id = :id LIMIT 1";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':id', $id, \PDO::PARAM_INT);
        $stmt->execute();
        $result = $stmt->fetch(\PDO::FETCH_ASSOC);
        if ($result && !empty($result['recommendations'])) {
            $result['recommendations'] = json_decode($result['recommendations'], true) ?? [];
        }
        return $result;
    }

    public function updateNotes($id, $notes) {
        $query = "UPDATE " . $this->table . " SET notes = :notes WHERE id = :id";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':notes', $notes);
        $stmt->bindParam(':id', $id, \PDO::PARAM_INT);
        return $stmt->execute();
    }

    public function archive($id) {
        $query = "UPDATE " . $this->table . " SET is_archived = TRUE WHERE id = :id";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':id', $id, \PDO::PARAM_INT);
        return $stmt->execute();
    }

    public function countByUserId($user_id) {
        $query = "SELECT COUNT(*) as count FROM " . $this->table . " WHERE user_id = :user_id AND is_archived = FALSE";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':user_id', $user_id, \PDO::PARAM_INT);
        $stmt->execute();
        $result = $stmt->fetch(\PDO::FETCH_ASSOC);
        return $result['count'] ?? 0;
    }

    public function getRiskDistribution($user_id) {
        $query = "SELECT risk_level, COUNT(*) as count FROM " . $this->table . " 
                  WHERE user_id = :user_id AND is_archived = FALSE AND status = 'COMPLETED' AND risk_level IS NOT NULL
                  GROUP BY risk_level";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':user_id', $user_id, \PDO::PARAM_INT);
        $stmt->execute();
        return $stmt->fetchAll(\PDO::FETCH_ASSOC);
    }

    public function getAllScans($limit = 100, $offset = 0) {
        $query = "SELECT s.*, u.email, u.first_name, u.last_name 
                  FROM " . $this->table . " s 
                  JOIN users u ON s.user_id = u.id 
                  ORDER BY s.created_at DESC 
                  LIMIT :limit OFFSET :offset";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':limit', $limit, \PDO::PARAM_INT);
        $stmt->bindParam(':offset', $offset, \PDO::PARAM_INT);
        $stmt->execute();
        return $stmt->fetchAll(\PDO::FETCH_ASSOC);
    }

    public function getScanStats() {
        $query = "SELECT 
                    COUNT(*) as total_scans,
                    SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) as high_risk,
                    SUM(CASE WHEN risk_level = 'Medium' THEN 1 ELSE 0 END) as medium_risk,
                    SUM(CASE WHEN risk_level = 'Low' THEN 1 ELSE 0 END) as low_risk,
                    COUNT(DISTINCT user_id) as unique_users
                  FROM " . $this->table . " 
                  WHERE is_archived = FALSE AND status = 'COMPLETED'";
        $stmt = $this->db->query($query);
        return $stmt->fetch(\PDO::FETCH_ASSOC);
    }

    public function getRecentScansByClassification($days = 30) {
        $query = "SELECT classification, COUNT(*) as count 
                  FROM " . $this->table . " 
                  WHERE created_at > DATE_SUB(NOW(), INTERVAL :days DAY) 
                  AND is_archived = FALSE AND status = 'COMPLETED' AND classification IS NOT NULL
                  GROUP BY classification 
                  ORDER BY count DESC";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':days', $days, \PDO::PARAM_INT);
        $stmt->execute();
        return $stmt->fetchAll(\PDO::FETCH_ASSOC);
    }
}
