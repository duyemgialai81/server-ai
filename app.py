import os
import io
import requests
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv

# 1. Cấu hình môi trường
load_dotenv()
app = Flask(__name__)
CORS(app) # Cho phép Spring Boot gọi sang

HF_TOKEN = os.getenv("HF_TOKEN")
# Model chuyên dụng để thay đổi vùng ảnh dựa trên mask (giữ nguyên vùng còn lại)
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"

def img_to_b64(image):
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ready", "token_configured": bool(HF_TOKEN)})

@app.route('/process-hair', methods=['POST'])
def process_hair():
    try:
        if not HF_TOKEN:
            return jsonify({"error": "Missing HF_TOKEN"}), 500

        data = request.json
        image_url = data.get('image_url')
        # Prompt mô tả kiểu tóc muốn đổi
        hair_style = data.get('prompt', "realistic modern hairstyle, highly detailed, professional hair salon")

        # 1. Tải ảnh gốc từ URL (Spring Boot gửi qua)
        response = requests.get(image_url, timeout=15)
        original_img = Image.open(io.BytesIO(response.content)).convert("RGB")
        original_img = ImageOps.fit(original_img, (512, 512)) # Chuẩn hóa kích thước AI

        # 2. Tạo Mask (Mặt nạ): Vùng trắng (AI vẽ lại), Vùng đen (Giữ nguyên mặt)
        mask = Image.new("L", (512, 512), 0)
        draw = ImageDraw.Draw(mask)
        # Vẽ vùng vẽ lại: Tóc thường ở phía trên (0-240) và hai bên
        draw.rectangle([0, 0, 512, 240], fill=255)  # Phần trên đầu
        draw.rectangle([0, 0, 110, 512], fill=255)  # Tóc mai bên trái
        draw.rectangle([402, 0, 512, 512], fill=255) # Tóc mai bên phải

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
        ai_response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if ai_response.status_code != 200:
            return jsonify({"error": "AI Model is starting up, please try again in 30s"}), 503

        # 4. Trả kết quả ảnh về cho Spring Boot
        return send_file(io.BytesIO(ai_response.content), mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)