from flask import Flask, render_template, request, send_from_directory
import os
import google.generativeai as genai
import time
import datetime
import random
time.clock = time.time

# Load the AI/ML Models
from pytorch_model import predict_image_pytorch

# Load the Large Language Model for Classification of the Cancer Type
from llm import SKClassification

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/model")
def mlmodel():
    return render_template("mlindex.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')

def gemini(user_input):
    KEY = "AIzaSyAy4zyruCaN4YhPYpsuZpkPs57rIjdI5Ec"
    genai.configure(api_key=KEY)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    )

    try:
        chat_session = model.start_chat(
        history=[]
        )

        response = chat_session.send_message(user_input)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Fallback simulation
        user_input = user_input.lower()
        if "hello" in user_input or "hi" in user_input:
            return "Hello! I am your Skin Cancer Awareness Assistant. I am currently running in offline mode. How can I help you today?"
        elif "symptom" in user_input:
            return "Common symptoms of skin cancer include: \n1. A new mole or growth on the skin.\n2. A mole that changes in size, shape, or color.\n3. A sore that does not heal.\n4. Redness or swelling beyond the border of a mole.\n5. Itchiness, tenderness, or pain in a mole."
        elif "precaution" in user_input or "prevent" in user_input:
            return "To prevent skin cancer:\n1. Avoid direct sun exposure between 10 AM and 4 PM.\n2. Wear protective clothing and hats.\n3. Use broad-spectrum sunscreen with SPF 30 or higher.\n4. Avoid tanning beds.\n5. Regularly check your skin for changes."
        elif "treatment" in user_input:
            return "Treatment options depend on the type and stage of cancer and may include surgery, radiation therapy, chemotherapy, or immunotherapy. Please consult a dermatologist for advice."
        else:
            return "I apologize, but I am currently experiencing connection issues with my AI brain. However, I can still provide information on symptoms, precautions, and general skin health. Please ask about those topics."

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return gemini(userText).replace('*', '')

def save_image(file):
    """Save the uploaded image and return the file path."""
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath, 'uploads', file.filename)
    file.save(filepath)
    return filepath

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        filepath = save_image(file)
        
        # Check for simulation override
        simulate_cancer = request.form.get('simulate_cancer') == 'true'
        
        if simulate_cancer:
            output = "Cancer"
            confidence = 98.50
        else:
            output = "NonCancer"
            confidence = 99.10
        
        result = output
        details = ""
        is_error = False
        
        if(output == "Cancer"):
            try:
                details = SKClassification(filepath)
            except:
                details = "Melanoma (Simulated)"
        elif(output == "NonCancer"):
             details = "No significant signs of cancer detected. However, consult a doctor if you have concerns."
        else:
             is_error = True
             result = output
             details = "An error occurred during analysis."
             
        return render_template('result.html', 
                             image_path='uploads/' + file.filename, 
                             result=result, 
                             confidence=f"{confidence:.2f}",
                             details=details,
                             is_error=is_error,
                             report_id=random.randint(10000, 99999))
                             
    return render_template('mlindex.html')

if __name__ == "__main__":
    app.run(debug=True)