# modules/controlcom_runner.py

import json
import subprocess
import tempfile
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "wrapper_controlcom.py"

def run_controlcom(description: str) -> list[dict]:
    """
    ControlCom으로 자연어 설명을 파싱해
    [
      {prompt:str, bbox:[x,y,w,h], yaw:float, pitch:float},
      ...
    ] 형태 리스트 반환
    """

    # 1. 임시 JSON 파일 생성
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        result_path = Path(tf.name)

    # 2. subprocess 실행
    cmd = [
        "python", str(SCRIPT_PATH),
        "--text", description,
        "--out", str(result_path),
    ]
    print(f"[DEBUG] ControlCom 실행: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 3. JSON 결과 파싱
    data = json.loads(result_path.read_text(encoding="utf-8"))
    print("[DEBUG] ControlCom 결과:", data)
    return data
