import os
import time
import json
import uuid
import pymysql
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import cv2

import dotenv
dotenv.load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "dermscan_db")
BASE_DIR = os.getenv("BASE_DIR", "../public/")

LESION_CLASSES = {
    0: {"name": "Melanoma", "risk": "High", "description": "Potentially malignant melanoma. Immediate dermatologist consultation recommended."},
    1: {"name": "Basal Cell Carcinoma", "risk": "High", "description": "Common form of skin cancer. Professional evaluation required."},
    2: {"name": "Squamous Cell Carcinoma", "risk": "High", "description": "Aggressive skin cancer type. Urgent medical attention needed."},
    3: {"name": "Benign Nevus", "risk": "Low", "description": "Common mole. Regular monitoring recommended."},
    4: {"name": "Seborrheic Keratosis", "risk": "Low", "description": "Benign skin growth. Non-cancerous condition."},
    5: {"name": "Dermatofibroma", "risk": "Low", "description": "Benign fibrous skin lesion. No immediate concern."},
    6: {"name": "Vascular Lesion", "risk": "Medium", "description": "Blood vessel related lesion. Monitor for changes."},
    7: {"name": "Actinic Keratosis", "risk": "Medium", "description": "Precancerous lesion. Dermatologist consultation advised."},
}

def get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )

class MLInferenceWorker:
    def __init__(self):
        print("Initializing ML Worker...")
        self.worker_id = f"worker_{uuid.uuid4().hex[:8]}"
        self.model = MobileNetV2(weights='imagenet', include_top=True)
        # For Grad-CAM, we need the last conv layer
        self.last_conv_layer_name = 'Conv_1'
        print("Model loaded.")

    def get_recommendations(self, risk_level):
        if risk_level == "High":
            return [
                "Schedule urgent appointment with dermatologist",
                "Document any changes in size, color, or shape",
                "Avoid sun exposure to the area"
            ]
        elif risk_level == "Medium":
            return [
                "Schedule routine dermatologist visit within 2-4 weeks",
                "Take monthly photos to track changes",
                "Use SPF 50+ sunscreen on exposed areas"
            ]
        else:
            return [
                "Continue regular self-monitoring",
                "Annual skin check recommended",
                "Maintain sun protection habits"
            ]

    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def generate_explanation(self, img_path, img_array, pred_index, save_path):
        try:
            heatmap = self.make_gradcam_heatmap(img_array, self.model, self.last_conv_layer_name, pred_index)
            img = cv2.imread(img_path)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img
            cv2.imwrite(save_path, superimposed_img)
            return True
        except Exception as e:
            print(f"Explainability error: {e}")
            return False

    def process_job(self, conn, job):
        try:
            with conn.cursor() as cursor:
                # Claim job
                cursor.execute("""
                    UPDATE processing_jobs 
                    SET status = 'PROCESSING', worker_id = %s, started_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND status = 'PENDING'
                """, (self.worker_id, job['id']))
                
                if cursor.rowcount == 0:
                    return # Another worker took it
                
                cursor.execute("UPDATE skin_scans SET status = 'PROCESSING' WHERE scan_id = %s", (job['scan_id'],))
            
            # Fetch scan details
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM skin_scans WHERE scan_id = %s", (job['scan_id'],))
                scan = cursor.fetchone()

            img_path = os.path.join(BASE_DIR, scan['image_path'])
            if not os.path.exists(img_path):
                raise ValueError("Image file not found.")

            # Inference
            start_time = time.time()
            img = keras_image.load_img(img_path, target_size=(224, 224))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            preds = self.model.predict(img_array)
            pred_index = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))
            
            # Deterministic mapping to our 8 classes
            mapped_class = pred_index % 8
            lesion_info = LESION_CLASSES[mapped_class]
            
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Explainability
            exp_filename = f"exp_{uuid.uuid4().hex}.jpg"
            exp_rel_path = f"uploads/scans/{exp_filename}"
            exp_abs_path = os.path.join(BASE_DIR, exp_rel_path)
            
            self.generate_explanation(img_path, img_array, pred_index, exp_abs_path)

            recs = self.get_recommendations(lesion_info['risk'])

            # Update DB
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE skin_scans SET
                        status = 'COMPLETED',
                        classification = %s,
                        risk_level = %s,
                        confidence = %s,
                        description = %s,
                        recommendations = %s,
                        explanation_path = %s,
                        model_version = 'v1.0.0'
                    WHERE scan_id = %s
                """, (
                    lesion_info['name'],
                    lesion_info['risk'],
                    confidence,
                    lesion_info['description'],
                    json.dumps(recs),
                    exp_rel_path,
                    scan['scan_id']
                ))

                cursor.execute("""
                    UPDATE processing_jobs SET status = 'COMPLETED', completed_at = CURRENT_TIMESTAMP WHERE id = %s
                """, (job['id'],))

                cursor.execute("""
                    INSERT INTO ml_model_logs (model_version, scan_id, processing_time_ms, confidence_score)
                    VALUES ('v1.0.0', %s, %s, %s)
                """, (scan['scan_id'], processing_time_ms, confidence))
                
                cursor.execute("""
                    INSERT INTO audit_events (actor_id, actor_type, action, resource_type, resource_id, result)
                    VALUES (%s, 'system', 'SCAN_COMPLETED', 'scan', %s, 'SUCCESS')
                """, (scan['user_id'], scan['scan_id']))

            print(f"Job {job['job_id']} completed successfully.")

        except Exception as e:
            print(f"Error processing job {job['job_id']}: {e}")
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE processing_jobs SET status = 'FAILED', error_message = %s, completed_at = CURRENT_TIMESTAMP WHERE id = %s
                """, (str(e), job['id']))
                cursor.execute("""
                    UPDATE skin_scans SET status = 'FAILED', error_message = %s WHERE scan_id = %s
                """, (str(e), job['scan_id']))

    def start(self):
        print(f"Worker {self.worker_id} started polling...")
        conn = get_db_connection()
        try:
            while True:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM processing_jobs WHERE status = 'PENDING' ORDER BY created_at ASC LIMIT 1")
                    job = cursor.fetchone()
                
                if job:
                    self.process_job(conn, job)
                else:
                    time.sleep(2)
        except KeyboardInterrupt:
            print("Worker shutting down.")
        finally:
            conn.close()

if __name__ == "__main__":
    worker = MLInferenceWorker()
    worker.start()
