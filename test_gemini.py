import google.generativeai as genai
import os

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

print("Testing Gemini API...")
try:
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message("Hello")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
