<?php
namespace App\Models;

require_once __DIR__ . '/../../config/database.php';

class Doctor {
    private $db;
    private $table = 'doctors';

    public function __construct() {
        $database = new \Database();
        $this->db = $database->connect();
    }

    public function findAll($city = null, $limit = 50) {
        $query = "SELECT * FROM " . $this->table . " WHERE is_active = TRUE";
        
        if ($city) {
            $query .= " AND city = :city";
        }
        
        $query .= " ORDER BY is_verified DESC, name ASC LIMIT :limit";
        
        $stmt = $this->db->prepare($query);
        
        if ($city) {
            $stmt->bindParam(':city', $city);
        }
        $stmt->bindParam(':limit', $limit, \PDO::PARAM_INT);
        
        $stmt->execute();
        return $stmt->fetchAll(\PDO::FETCH_ASSOC);
    }

    public function findById($id) {
        $query = "SELECT * FROM " . $this->table . " WHERE id = :id AND is_active = TRUE LIMIT 1";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':id', $id, \PDO::PARAM_INT);
        $stmt->execute();
        return $stmt->fetch(\PDO::FETCH_ASSOC);
    }

    public function getCities() {
        $query = "SELECT DISTINCT city FROM " . $this->table . " WHERE is_active = TRUE ORDER BY city";
        $stmt = $this->db->query($query);
        return $stmt->fetchAll(\PDO::FETCH_COLUMN);
    }

    public function create($data) {
        $query = "INSERT INTO " . $this->table . " 
                  (name, specialty, clinic_name, address, city, phone, email, website) 
                  VALUES 
                  (:name, :specialty, :clinic_name, :address, :city, :phone, :email, :website)";
        
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':name', $data['name']);
        $stmt->bindParam(':specialty', $data['specialty']);
        $stmt->bindParam(':clinic_name', $data['clinic_name']);
        $stmt->bindParam(':address', $data['address']);
        $stmt->bindParam(':city', $data['city']);
        $stmt->bindParam(':phone', $data['phone']);
        $stmt->bindParam(':email', $data['email']);
        $stmt->bindParam(':website', $data['website']);
        
        return $stmt->execute() ? $this->db->lastInsertId() : false;
    }

    public function update($id, $data) {
        $fields = [];
        foreach ($data as $key => $value) {
            $fields[] = "$key = :$key";
        }
        
        $query = "UPDATE " . $this->table . " SET " . implode(', ', $fields) . " WHERE id = :id";
        $stmt = $this->db->prepare($query);
        
        foreach ($data as $key => $value) {
            $stmt->bindValue(":" . $key, $value);
        }
        $stmt->bindParam(':id', $id, \PDO::PARAM_INT);
        
        return $stmt->execute();
    }

    public function delete($id) {
        $query = "UPDATE " . $this->table . " SET is_active = FALSE WHERE id = :id";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':id', $id, \PDO::PARAM_INT);
        return $stmt->execute();
    }
}
