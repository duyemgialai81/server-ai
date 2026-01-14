# app.py
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

# Models
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
        "hf_token_configured": bool(HF_TOKEN),
        "openai_configured": bool(OPENAI_API_KEY),
        "features": ["process-hair", "chat-advisor"]
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
            "parameters": {"num_inference_steps": 30}
        }

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        ai_response = requests.post(INPAINTING_MODEL, headers=headers, json=payload, timeout=60)
        
        if ai_response.status_code != 200:
            return jsonify({"error": "AI Model is starting up"}), 503

        return send_file(io.BytesIO(ai_response.content), mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat-advisor', methods=['POST'])
def chat_advisor():
    """
    AI Chatbot - Fallback to Hugging Face if no OpenAI key
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        hair_type = data.get('hair_type', 'normal')
        conversation_history = data.get('history', [])

        # ✅ SỬA: Fallback to Hugging Face nếu không có OpenAI
        if OPENAI_API_KEY:
            # Use OpenAI
            try:
                import openai
                openai.api_key = OPENAI_API_KEY
                
                system_prompt = f"""Bạn là chuyên gia tư vấn chăm sóc tóc tại Việt Nam.
Loại tóc khách hàng: {hair_type}

Trả lời bằng tiếng Việt, tư vấn sản phẩm cụ thể với giá và nơi mua."""

                messages = [{"role": "system", "content": system_prompt}]
                for msg in conversation_history[-10:]:
                    messages.append({"role": msg.get('role', 'user'), "content": msg.get('content', '')})
                messages.append({"role": "user", "content": user_message})

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )

                ai_text = response.choices[0].message.content.strip()
                return jsonify({
                    "response": ai_text,
                    "model_loading": False,
                    "hair_type": hair_type,
                    "model_used": "gpt-3.5-turbo"
                })
            except Exception as openai_error:
                print(f"OpenAI error: {openai_error}, falling back to Hugging Face")
                # Fall through to Hugging Face
        
        # ✅ Fallback: Use Hugging Face (FREE)
        if not HF_TOKEN:
            return jsonify({"error": "Missing HF_TOKEN"}), 500

        system_prompt = f"""You are a professional hair care advisor in Vietnam.
User's hair type: {hair_type}
Always respond in Vietnamese language with specific product recommendations."""

        conversation = f"{system_prompt}\n\n"
        for msg in conversation_history[-5:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            conversation += f"{'User' if role == 'user' else 'Assistant'}: {content}\n"
        conversation += f"User: {user_message}\nAssistant:"

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": conversation,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.95,
                "return_full_text": False
            }
        }

        ai_response = requests.post(CHAT_MODEL, headers=headers, json=payload, timeout=30)
        
        if ai_response.status_code == 503:
            return jsonify({
                "response": "AI đang khởi động, vui lòng thử lại sau 20 giây.",
                "model_loading": True
            }), 200
        
        if ai_response.status_code != 200:
            return jsonify({"error": "AI service unavailable"}), 503

        result = ai_response.json()
        if isinstance(result, list) and len(result) > 0:
            ai_text = result[0].get('generated_text', '').strip()
        else:
            ai_text = "Xin lỗi, tôi không thể trả lời lúc này."

        return jsonify({
            "response": ai_text,
            "model_loading": False,
            "hair_type": hair_type,
            "model_used": "mistral-7b"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze-hair', methods=['POST'])
def analyze_hair():
    """Phân tích tóc - Mock data nếu không có Vision API"""
    try:
        data = request.json
        image_url = data.get('image_url')

        analysis = {
            "hair_condition": "healthy",
            "recommendations": [
                "Sử dụng dầu gội dưỡng ẩm",
                "Dùng mặt nạ ủ tóc 2 lần/tuần",
                "Tránh sấy tóc nhiệt độ cao"
            ],
            "suggested_products": [
                {
                    "name": "Dầu gội Tresemmé Keratin Smooth",
                    "type": "shampoo",
                    "where_to_buy": "Hasaki, Guardian",
                    "price_range": "150,000 - 200,000 VNĐ"
                }
            ],
            "salon_services": [
                "Phục hồi tóc hư tổn",
                "Cắt tỉa đuôi tóc"
            ]
        }

        return jsonify(analysis)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)