<?php
namespace App\Models;

require_once __DIR__ . '/../../config/database.php';

class User {
    private $db;
    private $table = 'users';

    public $id;
    public $email;
    public $first_name;
    public $last_name;
    public $age;
    public $skin_type;
    public $family_history;
    public $privacy_opt_in;
    public $is_active;
    public $is_admin;
    public $created_at;

    public function __construct() {
        $database = new \Database();
        $this->db = $database->connect();
    }

    public function findByEmail($email) {
        $query = "SELECT * FROM " . $this->table . " WHERE email = :email LIMIT 1";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':email', $email);
        $stmt->execute();
        return $stmt->fetch(\PDO::FETCH_ASSOC);
    }

    public function findById($id) {
        $query = "SELECT id, email, first_name, last_name, age, skin_type, family_history, 
                  privacy_opt_in, is_active, is_admin, created_at 
                  FROM " . $this->table . " WHERE id = :id LIMIT 1";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':id', $id, \PDO::PARAM_INT);
        $stmt->execute();
        return $stmt->fetch(\PDO::FETCH_ASSOC);
    }

    public function create($data) {
        $query = "INSERT INTO " . $this->table . " 
                  (email, password_hash, first_name, last_name, age, skin_type, family_history, privacy_opt_in, email_verified_at) 
                  VALUES 
                  (:email, :password_hash, :first_name, :last_name, :age, :skin_type, :family_history, :privacy_opt_in, NOW())";

        $stmt = $this->db->prepare($query);
        
        // Hash password
        $password_hash = password_hash($data['password'], PASSWORD_BCRYPT);

        $stmt->bindParam(':email', $data['email']);
        $stmt->bindParam(':password_hash', $password_hash);
        $stmt->bindParam(':first_name', $data['first_name']);
        $stmt->bindParam(':last_name', $data['last_name']);
        $stmt->bindParam(':age', $data['age'], \PDO::PARAM_INT);
        $stmt->bindParam(':skin_type', $data['skin_type']);
        $stmt->bindParam(':family_history', $data['family_history'], \PDO::PARAM_BOOL);
        $stmt->bindParam(':privacy_opt_in', $data['privacy_opt_in'], \PDO::PARAM_BOOL);

        if ($stmt->execute()) {
            return $this->db->lastInsertId();
        }
        return false;
    }

    public function update($id, $data) {
        $fields = [];
        $params = [':id' => $id];

        if (isset($data['first_name'])) {
            $fields[] = "first_name = :first_name";
            $params[':first_name'] = $data['first_name'];
        }
        if (isset($data['last_name'])) {
            $fields[] = "last_name = :last_name";
            $params[':last_name'] = $data['last_name'];
        }
        if (isset($data['age'])) {
            $fields[] = "age = :age";
            $params[':age'] = $data['age'];
        }
        if (isset($data['skin_type'])) {
            $fields[] = "skin_type = :skin_type";
            $params[':skin_type'] = $data['skin_type'];
        }
        if (isset($data['family_history'])) {
            $fields[] = "family_history = :family_history";
            $params[':family_history'] = $data['family_history'];
        }
        if (isset($data['privacy_opt_in'])) {
            $fields[] = "privacy_opt_in = :privacy_opt_in";
            $params[':privacy_opt_in'] = $data['privacy_opt_in'];
        }

        if (empty($fields)) {
            return false;
        }

        $query = "UPDATE " . $this->table . " SET " . implode(', ', $fields) . " WHERE id = :id";
        $stmt = $this->db->prepare($query);

        foreach ($params as $key => $value) {
            $stmt->bindValue($key, $value);
        }

        return $stmt->execute();
    }

    public function updateLastLogin($id) {
        $query = "UPDATE " . $this->table . " SET last_login_at = NOW() WHERE id = :id";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':id', $id, \PDO::PARAM_INT);
        return $stmt->execute();
    }

    public function verifyPassword($password, $hash) {
        return password_verify($password, $hash);
    }

    public function recordLoginAttempt($email, $ip_address, $success) {
        $query = "INSERT INTO login_attempts (email, ip_address, is_successful) VALUES (:email, :ip, :success)";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':email', $email);
        $stmt->bindParam(':ip', $ip_address);
        $stmt->bindParam(':success', $success, \PDO::PARAM_BOOL);
        return $stmt->execute();
    }

    public function getRecentFailedAttempts($email, $ip_address, $minutes = 15) {
        $query = "SELECT COUNT(*) as count FROM login_attempts 
                  WHERE (email = :email OR ip_address = :ip) 
                  AND is_successful = FALSE 
                  AND attempted_at > DATE_SUB(NOW(), INTERVAL :minutes MINUTE)";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':email', $email);
        $stmt->bindParam(':ip', $ip_address);
        $stmt->bindParam(':minutes', $minutes, \PDO::PARAM_INT);
        $stmt->execute();
        $result = $stmt->fetch(\PDO::FETCH_ASSOC);
        return $result['count'] ?? 0;
    }

    public function getAllUsers($limit = 50, $offset = 0) {
        $query = "SELECT id, email, first_name, last_name, is_active, is_admin, created_at, last_login_at 
                  FROM " . $this->table . " 
                  WHERE is_admin = FALSE 
                  ORDER BY created_at DESC 
                  LIMIT :limit OFFSET :offset";
        $stmt = $this->db->prepare($query);
        $stmt->bindParam(':limit', $limit, \PDO::PARAM_INT);
        $stmt->bindParam(':offset', $offset, \PDO::PARAM_INT);
        $stmt->execute();
        return $stmt->fetchAll(\PDO::FETCH_ASSOC);
    }

    public function countAllUsers() {
        $query = "SELECT COUNT(*) as count FROM " . $this->table . " WHERE is_admin = FALSE";
        $stmt = $this->db->query($query);
        $result = $stmt->fetch(\PDO::FETCH_ASSOC);
        return $result['count'] ?? 0;
    }
}
