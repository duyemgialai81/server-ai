import os
import io
import requests
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv
from openai import OpenAI  # ✅ Cập nhật cho thư viện OpenAI mới

load_dotenv()
app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Khởi tạo client OpenAI (Chỉ khởi tạo nếu có Key)
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Models
INPAINTING_MODEL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"
# Link Models đúng chuẩn 2026
CHAT_MODEL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"

def img_to_b64(image):
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

# ✅ Sửa thành POST để Postman của bạn không bị lỗi 405 nữa
@app.route('/health', methods=['GET', 'POST'])
def health():
    return jsonify({
        "status": "ready", 
        "hf_token_configured": bool(HF_TOKEN),
        "openai_configured": bool(OPENAI_API_KEY)
    })

@app.route('/process-hair', methods=['POST'])
def process_hair():
    try:
        if not HF_TOKEN:
            return jsonify({"error": "Missing HF_TOKEN"}), 500
        
        data = request.json
        image_url = data.get('image_url')
        hair_style = data.get('prompt', "realistic modern hairstyle")

        response = requests.get(image_url, timeout=15)
        original_img = Image.open(io.BytesIO(response.content)).convert("RGB")
        original_img = ImageOps.fit(original_img, (512, 512))

        mask = Image.new("L", (512, 512), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([0, 0, 512, 240], fill=255)
        draw.rectangle([0, 0, 110, 512], fill=255)
        draw.rectangle([402, 0, 512, 512], fill=255)

        payload = {
            "inputs": {
                "image": img_to_b64(original_img),
                "mask_image": img_to_b64(mask),
                "prompt": f"{hair_style}, natural look, photorealistic, 8k",
                "negative_prompt": "deformed face, ugly, changed eyes"
            },
            "parameters": {"num_inference_steps": 30},
            "options": {"wait_for_model": True} # ✅ Khắc phục 503
        }

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        ai_response = requests.post(INPAINTING_MODEL, headers=headers, json=payload, timeout=90)
        
        if ai_response.status_code != 200:
            return jsonify({"error": "AI Model is starting up"}), 503

        return send_file(io.BytesIO(ai_response.content), mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat-advisor', methods=['POST'])
def chat_advisor():
    try:
        data = request.json
        user_message = data.get('message', '')
        hair_type = data.get('hair_type', 'normal')
        conversation_history = data.get('history', [])

        # ✅ Cập nhật OpenAI logic cho bản 1.0.0+
        if client:
            try:
                system_prompt = f"Bạn là chuyên gia tư vấn tóc tại Việt Nam. Loại tóc: {hair_type}."
                messages = [{"role": "system", "content": system_prompt}]
                
                # Chuyển đổi lịch sử chat phù hợp định dạng OpenAI
                for msg in conversation_history[-5:]:
                    messages.append({"role": msg.get('role', 'user'), "content": msg.get('content', '')})
                
                messages.append({"role": "user", "content": user_message})

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7
                )
                return jsonify({
                    "response": response.choices[0].message.content,
                    "model_used": "gpt-3.5-turbo"
                })
            except Exception as e:
                print(f"OpenAI error: {e}, falling back to Hugging Face")

        # ✅ Fallback: Hugging Face (Sửa lỗi 503 bằng wait_for_model)
        if not HF_TOKEN:
            return jsonify({"error": "No AI configuration found"}), 500

        prompt = f"<s>[INST] Bạn là chuyên gia tư vấn tóc. Trả lời bằng tiếng Việt: {user_message} [/INST]"
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 500, "temperature": 0.7},
            "options": {"wait_for_model": True} # ✅ Ép HF phải đợi model khởi động
        }

        hf_res = requests.post(CHAT_MODEL, headers=headers, json=payload, timeout=60)
        
        if hf_res.status_code == 200:
            result = hf_res.json()
            text = result[0]['generated_text'].split("[/INST]")[-1].strip()
            return jsonify({"response": text, "model_used": "mistral-7b"})
        
        return jsonify({"error": "AI Service Unavailable", "details": hf_res.text}), 503

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)