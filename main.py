# main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
from .pipeline import run_interior_pipeline

app = FastAPI()

@app.post("/interior/compose")
async def compose(
    file: UploadFile,
    description: str = Form(...)
):
    # 업로드된 이미지 저장
    uid = str(uuid.uuid4())[:8]
    input_path = Path(f"./assets/input_{uid}.jpg")
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 파이프라인 실행
    print(f"[*] 인테리어 파이프라인 시작: {description}")
    output_path = run_interior_pipeline(description, input_path)

    return FileResponse(output_path, media_type="image/jpeg")
