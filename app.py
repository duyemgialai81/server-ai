import os
import io
import requests
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv
from openai import OpenAI

# Tải biến môi trường
load_dotenv()
app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Khởi tạo client OpenAI mới nhất
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ✅ SỬ DỤNG ENDPOINT TRỰC TIẾP (Bỏ qua Router nếu tài khoản bị giới hạn quyền)
INPAINTING_MODEL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"
CHAT_MODEL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

def img_to_b64(image):
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.route('/health', methods=['GET', 'POST'])
def health():
    return jsonify({
        "status": "ready", 
        "hf_token_configured": bool(HF_TOKEN),
        "openai_configured": bool(OPENAI_API_KEY),
        "api_version": "2026.1.14"
    })

@app.route('/process-hair', methods=['POST'])
def process_hair():
    try:
        if not HF_TOKEN:
            return jsonify({"error": "Missing HF_TOKEN"}), 500
        
        data = request.json
        image_url = data.get('image_url')
        hair_style = data.get('prompt', "realistic modern hairstyle")

        # Xử lý ảnh đầu vào
        response = requests.get(image_url, timeout=15)
        original_img = Image.open(io.BytesIO(response.content)).convert("RGB")
        original_img = ImageOps.fit(original_img, (512, 512))

        # Tạo Mask thông minh
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

@app.route('/chat-advisor', methods=['POST'])
def chat_advisor():
    try:
        data = request.json
        user_message = data.get('message', '')
        hair_type = data.get('hair_type', 'normal')
        history = data.get('history', [])

        # 1. Thử sử dụng OpenAI trước
        if client:
            try:
                system_prompt = f"Bạn là chuyên gia tư vấn tóc chuyên nghiệp tại Việt Nam. Loại tóc khách hàng: {hair_type}."
                messages = [{"role": "system", "content": system_prompt}]
                for msg in history[-5:]:
                    messages.append({"role": msg.get('role', 'user'), "content": msg.get('content', '')})
                messages.append({"role": "user", "content": user_message})

                chat_res = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7
                )
                return jsonify({
                    "response": chat_res.choices[0].message.content,
                    "model_used": "gpt-3.5-turbo"
                })
            except Exception as e:
                print(f"OpenAI fallback: {e}")

        # 2. Sử dụng Hugging Face trực tiếp (Tránh lỗi phân quyền Router)
        if not HF_TOKEN:
            return jsonify({"error": "AI Config Missing"}), 500

        # Cấu trúc Prompt Mistral
        prompt = f"<s>[INST] Bạn là chuyên gia tư vấn tóc Việt Nam. Trả lời bằng tiếng Việt câu hỏi sau cho khách có tóc {hair_type}: {user_message} [/INST]"
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 600, "temperature": 0.7},
            "options": {"wait_for_model": True, "use_cache": False} # use_cache: False giúp tránh lấy kết quả cũ lỗi
        }

        hf_res = requests.post(CHAT_MODEL, headers=headers, json=payload, timeout=60)
        
        if hf_res.status_code == 200:
            result = hf_res.json()
            # Xử lý chuỗi để lấy phản hồi sạch
            full_text = result[0].get('generated_text', '')
            clean_text = full_text.split("[/INST]")[-1].strip()
            return jsonify({"response": clean_text, "model_used": "mistral-7b"})
        
        # Xử lý lỗi 503 chi tiết
        return jsonify({
            "response": "AI đang bận khởi động hệ thống. Vui lòng nhắn lại sau 30 giây.",
            "error": "503_BUSY",
            "details": hf_res.text
        }), 503

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Render yêu cầu chạy trên port được cấp
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)