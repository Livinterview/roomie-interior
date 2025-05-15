import os
import json
import base64
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"

def rotate_with_zero123(image_path: str, yaw: float, pitch: float, object_id: str) -> str:
    # 이미지 base64 인코딩
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "input": {
            "image_base64": encoded_image,
            "yaw": float(yaw),
            "pitch": float(pitch),
            "size": 256,
            "object_id": object_id
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    res = requests.post(API_URL, headers=headers, json=payload)
    print("[DEBUG] RunPod Response:", res.status_code, res.text)
    res.raise_for_status()

    job_id = res.json()["id"]

    # 결과 대기
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    print(f"[INFO] RunPod 작업 시작 - Job ID: {job_id}")
    print(f"[INFO] 결과를 기다리는 중입니다...")

    while True:
        result = requests.get(status_url, headers=headers).json()
        status = result["status"]
        print(f"[DEBUG] 현재 상태: {status}")

        if status == "COMPLETED":
            print("[INFO] RunPod 작업 완료 ✅")
            break
        elif status == "FAILED":
            raise RuntimeError("❌ RunPod 작업 실패")
        
        time.sleep(2)  # 2초마다 체크


    # 결과 이미지 다운로드
    output_url = result["output"]["url"]
    output_path = Path(f"interior/assets/{object_id}_rotated.png")
    img = requests.get(output_url)
    with open(output_path, "wb") as f:
        f.write(img.content)

    return str(output_path)
