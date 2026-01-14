import os
import io
import requests
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv
import openai  # Thêm OpenAI

# 1. Cấu hình môi trường
load_dotenv()
app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Thêm OpenAI key

# Configure OpenAI
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Models
INPAINTING_MODEL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"

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
        "features": ["process-hair", "chat-advisor", "analyze-hair"]
    })

@app.route('/process-hair', methods=['POST'])
def process_hair():
    try:
        if not HF_TOKEN:
            return jsonify({"error": "Missing HF_TOKEN"}), 500

        data = request.json
        image_url = data.get('image_url')
        hair_style = data.get('prompt', "realistic modern hairstyle, highly detailed, professional hair salon")

        # 1. Tải ảnh gốc từ URL
        response = requests.get(image_url, timeout=15)
        original_img = Image.open(io.BytesIO(response.content)).convert("RGB")
        original_img = ImageOps.fit(original_img, (512, 512))

        # 2. Tạo Mask
        mask = Image.new("L", (512, 512), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([0, 0, 512, 240], fill=255)
        draw.rectangle([0, 0, 110, 512], fill=255)
        draw.rectangle([402, 0, 512, 512], fill=255)

        # 3. Gửi yêu cầu tới Hugging Face API
        payload = {
            "inputs": {
                "image": img_to_b64(original_img),
                "mask_image": img_to_b64(mask),
                "prompt": f"{hair_style}, natural look, photorealistic, 8k",
                "negative_prompt": "deformed face, ugly, changed eyes, distorted facial features, blurry"
            },
            "parameters": {"num_inference_steps": 30}
        }

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        ai_response = requests.post(INPAINTING_MODEL, headers=headers, json=payload, timeout=60)
        
        if ai_response.status_code != 200:
            return jsonify({"error": "AI Model is starting up, please try again in 30s"}), 503

        return send_file(io.BytesIO(ai_response.content), mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat-advisor', methods=['POST'])
def chat_advisor():
    """
    AI Chatbot tư vấn chăm sóc tóc - Dùng OpenAI GPT-3.5/GPT-4
    """
    try:
        if not OPENAI_API_KEY:
            return jsonify({"error": "Missing OPENAI_API_KEY"}), 500

        data = request.json
        user_message = data.get('message', '')
        hair_type = data.get('hair_type', 'normal')
        conversation_history = data.get('history', [])

        # Tạo system prompt chuyên nghiệp
        system_prompt = f"""Bạn là chuyên gia tư vấn chăm sóc tóc chuyên nghiệp tại Việt Nam.

Loại tóc của khách hàng: {hair_type}

Chuyên môn của bạn:
- Tư vấn sản phẩm: dầu gội, dầu xả, serum, mặt nạ ủ tóc
- Hướng dẫn quy trình chăm sóc tóc hàng ngày
- Giải quyết các vấn đề về tóc: gàu, rụng tóc, khô xơ, dầu
- Đề xuất thương hiệu uy tín tại Việt Nam
- Gợi ý địa điểm mua hàng chính hãng

Yêu cầu:
1. Trả lời LUÔN bằng tiếng Việt
2. Cụ thể về tên sản phẩm, thương hiệu, giá cả
3. Nêu rõ nơi mua uy tín: Hasaki, Guardian, Watson's, Shopee Mall, Lazada Mall
4. Thân thiện, chuyên nghiệp, dễ hiểu
5. Đưa ra lời khuyên thực tế, có thể áp dụng ngay

Nếu không chắc chắn, hãy đưa ra 2-3 lựa chọn và giải thích ưu nhược điểm."""

        # Build messages cho OpenAI
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Last 10 messages
            messages.append({
                "role": msg.get('role', 'user'),
                "content": msg.get('content', '')
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": user_message
        })

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Hoặc "gpt-4" nếu có access
            messages=messages,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0.3,
            presence_penalty=0.3
        )

        ai_text = response.choices[0].message.content.strip()

        return jsonify({
            "response": ai_text,
            "model_loading": False,
            "hair_type": hair_type,
            "model_used": "gpt-3.5-turbo"
        })

    except openai.error.RateLimitError:
        return jsonify({
            "response": "Hiện tại có quá nhiều người dùng. Vui lòng thử lại sau 10 giây.",
            "model_loading": True
        }), 200
    except openai.error.APIError as e:
        return jsonify({
            "response": f"Lỗi kết nối AI: {str(e)}. Vui lòng thử lại.",
            "error": True
        }), 500
    except Exception as e:
        return jsonify({
            "response": "Xin lỗi, tôi không thể trả lời lúc này. Vui lòng thử lại sau.",
            "error": str(e)
        }), 500


@app.route('/analyze-hair', methods=['POST'])
def analyze_hair():
    """
    Phân tích tình trạng tóc từ ảnh - Dùng OpenAI Vision (GPT-4 Vision)
    """
    try:
        if not OPENAI_API_KEY:
            return jsonify({"error": "Missing OPENAI_API_KEY"}), 500

        data = request.json
        image_url = data.get('image_url')

        # Sử dụng GPT-4 Vision để phân tích ảnh
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": """Bạn là chuyên gia phân tích tóc. Hãy phân tích tình trạng tóc từ ảnh và đưa ra:
1. Tình trạng tóc (khỏe mạnh/khô/dầu/hư tổn)
2. 3-5 lời khuyên cụ thể
3. 2-3 sản phẩm phù hợp với giá cả và nơi mua tại Việt Nam
4. Dịch vụ salon nên dùng

Trả lời bằng JSON với format:
{
  "hair_condition": "...",
  "recommendations": [...],
  "suggested_products": [{name, type, where_to_buy, price_range}],
  "salon_services": [...]
}"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hãy phân tích tình trạng tóc trong ảnh này và đưa ra khuyến nghị."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.5
        )

        ai_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        # Remove markdown code blocks if present
        if "```json" in ai_text:
            ai_text = ai_text.split("```json")[1].split("```")[0].strip()
        elif "```" in ai_text:
            ai_text = ai_text.split("```")[1].split("```")[0].strip()
        
        analysis = json.loads(ai_text)
        
        return jsonify(analysis)

    except json.JSONDecodeError:
        # Fallback nếu AI không trả về JSON đúng format
        return jsonify({
            "hair_condition": "normal",
            "recommendations": [
                "Sử dụng dầu gội phù hợp với loại tóc",
                "Massage da đầu đều đặn",
                "Tránh nhiệt độ cao khi sấy tóc"
            ],
            "suggested_products": [
                {
                    "name": "Dầu gội Tresemmé Keratin Smooth",
                    "type": "shampoo",
                    "where_to_buy": "Hasaki, Guardian, Watson's",
                    "price_range": "150,000 - 200,000 VNĐ"
                }
            ],
            "salon_services": [
                "Phục hồi tóc hư tổn",
                "Cắt tỉa đuôi tóc"
            ],
            "note": "Phân tích tự động có thể không chính xác 100%. Nên tham khảo thêm chuyên gia."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)