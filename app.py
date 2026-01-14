import os
import io
import requests
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq  # ✅ Thư viện mới cho chat nhanh

# Tải biến môi trường
load_dotenv()
app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # ✅ Lấy Key từ ENV

# Khởi tạo các AI Clients
client_openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
client_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Models
INPAINTING_MODEL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"

def img_to_b64(image):
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.route('/health', methods=['GET', 'POST'])
def health():
    return jsonify({
        "status": "ready", 
        "hf_configured": bool(HF_TOKEN),
        "openai_configured": bool(OPENAI_API_KEY),
        "groq_configured": bool(GROQ_API_KEY)
    })

# --- PHẦN 1: TẠO KIỂU TÓC (Dùng Stable Diffusion - Hugging Face) ---
@app.route('/process-hair', methods=['POST'])
def process_hair():
    try:
        if not HF_TOKEN:
            return jsonify({"error": "Missing HF_TOKEN"}), 500
        
        data = request.json
        image_url = data.get('image_url')
        hair_style = data.get('prompt', "realistic modern hairstyle")

        # Tải và xử lý ảnh
        response = requests.get(image_url, timeout=15)
        original_img = Image.open(io.BytesIO(response.content)).convert("RGB")
        original_img = ImageOps.fit(original_img, (512, 512))

        # Tạo Mask vùng tóc (Che phần đầu và hai bên)
        mask = Image.new("L", (512, 512), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([0, 0, 512, 240], fill=255)
        draw.rectangle([0, 0, 110, 512], fill=255)
        draw.rectangle([402, 0, 512, 512], fill=255)

        payload = {
            "inputs": {
                "image": img_to_b64(original_img),
                "mask_image": img_to_b64(mask),
                "prompt": f"{hair_style}, high quality hairstyle, photorealistic, 8k",
                "negative_prompt": "blurry, deformed face, bad anatomy, ugly eyes"
            },
            "parameters": {"num_inference_steps": 30},
            "options": {"wait_for_model": True}
        }

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        ai_response = requests.post(INPAINTING_MODEL, headers=headers, json=payload, timeout=95)
        
        if ai_response.status_code == 200:
            return send_file(io.BytesIO(ai_response.content), mimetype='image/png')
        
        return jsonify({"error": "AI image service busy", "status": ai_response.status_code}), 503

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- PHẦN 2: TƯ VẤN CHĂM SÓC TÓC (Ưu tiên GROQ siêu tốc) ---
@app.route('/chat-advisor', methods=['POST'])
def chat_advisor():
    try:
        data = request.json
        user_message = data.get('message', '')
        hair_type = data.get('hair_type', 'normal')
        history = data.get('history', [])

        # 1. ƯU TIÊN SỐ 1: Dùng GROQ (Nhanh nhất, miễn phí ổn định)
        if client_groq:
            try:
                system_prompt = f"Bạn là chuyên gia tư vấn tóc chuyên nghiệp tại Việt Nam. Loại tóc khách hàng: {hair_type}. Trả lời thân thiện bằng tiếng Việt."
                messages = [{"role": "system", "content": system_prompt}]
                
                # Thêm lịch sử chat
                for msg in history[-5:]:
                    messages.append({"role": msg.get('role', 'user'), "content": msg.get('content', '')})
                
                messages.append({"role": "user", "content": user_message})

                chat_res = client_groq.chat.completions.create(
                    model="llama-3.3-70b-versatile", # Model mạnh nhất của Groq hiện tại
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )
                return jsonify({
                    "response": chat_res.choices[0].message.content,
                    "model_used": "groq-llama-3.3"
                })
            except Exception as e:
                print(f"Groq Error: {e}, falling back to OpenAI/HF")

        # 2. DỰ PHÒNG 1: Dùng OpenAI nếu Groq lỗi
        if client_openai:
            try:
                # ... (logic OpenAI tương tự)
                res = client_openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": user_message}]
                )
                return jsonify({"response": res.choices[0].message.content, "model_used": "gpt-3.5"})
            except: pass

        return jsonify({"error": "Tất cả dịch vụ AI đang bận"}), 503

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)