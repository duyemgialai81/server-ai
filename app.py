import os
import io
import requests
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Link Models
INPAINTING_MODEL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"
CHAT_MODEL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

def img_to_b64(image):
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ready", 
        "hf_configured": bool(HF_TOKEN),
        "openai_configured": bool(OPENAI_API_KEY)
    })

# 1. CHỨC NĂNG TẠO KIỂU TÓC (IMAGE AI)
@app.route('/process-hair', methods=['POST'])
def process_hair():
    try:
        if not HF_TOKEN:
            return jsonify({"error": "Thiếu HF_TOKEN"}), 500

        data = request.json
        image_url = data.get('image_url')
        hair_style = data.get('prompt', "realistic modern hairstyle")

        # Tải ảnh từ URL
        response = requests.get(image_url, timeout=15)
        original_img = Image.open(io.BytesIO(response.content)).convert("RGB")
        original_img = ImageOps.fit(original_img, (512, 512))

        # Tạo mặt nạ (Mask) vùng tóc
        mask = Image.new("L", (512, 512), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([0, 0, 512, 240], fill=255) # Vùng trên đầu
        draw.rectangle([0, 0, 110, 512], fill=255) # Vùng bên trái
        draw.rectangle([402, 0, 512, 512], fill=255) # Vùng bên phải

        payload = {
            "inputs": {
                "image": img_to_b64(original_img),
                "mask_image": img_to_b64(mask),
                "prompt": f"{hair_style}, natural look, photorealistic, 8k",
                "negative_prompt": "deformed face, ugly, changed eyes, bad anatomy"
            },
            "parameters": {"num_inference_steps": 30},
            "options": {"wait_for_model": True} # ✅ QUAN TRỌNG: Đợi model khởi động
        }

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        ai_response = requests.post(INPAINTING_MODEL, headers=headers, json=payload, timeout=90)
        
        if ai_response.status_code != 200:
            return jsonify({"error": "AI tạo ảnh đang khởi động, vui lòng thử lại sau 30s"}), 503

        return send_file(io.BytesIO(ai_response.content), mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2. CHỨC NĂNG TƯ VẤN CHĂM SÓC TÓC (CHAT AI)
@app.route('/chat-advisor', methods=['POST'])
def chat_advisor():
    try:
        data = request.json
        user_message = data.get('message', '')
        hair_type = data.get('hair_type', 'normal')
        history = data.get('history', [])

        # Ưu tiên sử dụng OpenAI nếu có Key
        if OPENAI_API_KEY:
            try:
                import openai
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                messages = [{"role": "system", "content": f"Bạn là chuyên gia tư vấn tóc. Loại tóc: {hair_type}. Trả lời bằng tiếng Việt."}]
                for m in history[-5:]: messages.append(m)
                messages.append({"role": "user", "content": user_message})
                
                res = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
                return jsonify({"response": res.choices[0].message.content, "model": "gpt-3.5"})
            except: pass 

        # Dự phòng: Sử dụng Hugging Face (Mistral)
        if not HF_TOKEN:
            return jsonify({"error": "Thiếu token AI"}), 500

        # Format prompt cho Mistral 7B
        prompt = f"<s>[INST] Bạn là chuyên gia tóc tại Việt Nam. Khách có tóc {hair_type}. Trả lời ngắn gọn bằng tiếng Việt: {user_message} [/INST]"

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 500, "temperature": 0.7},
            "options": {"wait_for_model": True} # ✅ QUAN TRỌNG: Đợi model khởi động
        }

        ai_response = requests.post(CHAT_MODEL, headers=headers, json=payload, timeout=60)
        
        if ai_response.status_code == 200:
            result = ai_response.json()
            full_text = result[0].get('generated_text', '')
            # Lấy phần trả lời sau thẻ [/INST]
            clean_text = full_text.split("[/INST]")[-1].strip()
            return jsonify({"response": clean_text, "model": "mistral-7b"})
        
        return jsonify({"response": "AI đang khởi động, hãy nhắn lại sau 20 giây.", "loading": True}), 503

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)