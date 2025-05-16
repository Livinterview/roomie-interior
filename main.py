from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from pathlib import Path
import uuid

from pipeline import run_interior_pipeline

app = FastAPI()

# ---------------------------------------------------------------
# /interior/compose  엔드포인트
#   • file          : 빈 방 이미지 (multipart/form-data)
#   • description   : 사용자의 대화 요약(가구 배치 요구)
#   • room_summary  : GPT-Vision 등으로 얻은 방 구조 설명
# ---------------------------------------------------------------

@app.post("/interior/compose")
async def compose(
    file: UploadFile,
    description: str = Form(...),
    room_summary: str | None = Form(None),
):
    # 1) 업로드 이미지 임시 저장 -----------------------------------
    uid = uuid.uuid4().hex[:8]
    input_path = Path(f"./assets/input_{uid}.jpg")
    input_path.parent.mkdir(exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 2) 파이프라인 실행 ------------------------------------------
    print("[*] 인테리어 파이프라인 시작")
    output_path = run_interior_pipeline(
        description=description,
        image_path=input_path,
        room_summary=room_summary or "",
    )

    # 3) 결과 이미지 반환 ------------------------------------------
    return FileResponse(output_path, media_type="image/jpeg")
