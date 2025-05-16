import subprocess, os
from pathlib import Path
from huggingface_hub import hf_hub_download

# 0) venv38 Python 경로 확인 ────────────────────────────
python_path = "/workspace/venv38/bin/python"
if not os.path.exists(python_path):
    raise RuntimeError("venv38 Python not found.")

# ──────────────────────────────────────────────────────
def rotate_with_zero123(image_path: str, yaw: float, pitch: float, object_id: str) -> str:

    # 1) 체크포인트는 캐시에 한 번만 내려받음
    ckpt_path = hf_hub_download(
        repo_id="cvlab/zero123-weights",
        filename="105000.ckpt"
    )

    # 2) 경로·이름 세팅
    config_path  = "/workspace/roomie-interior/external/zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml"
    script_path  = "/workspace/roomie-interior/infer_zero123.py"
    out_dir      = Path("assets")                       # 수정 (폴더 통일)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path     = out_dir / f"{object_id}_rot.png"

    # 3) 서브프로세스 커맨드
    command = [
        python_path,  script_path,
        "--input",  image_path,                  # 수정 (변수명 일치)
        "--yaw",    str(yaw),
        "--pitch",  str(pitch),
        "--output", str(out_path),               # 수정 (전체 경로 전달)
        "--checkpoint", ckpt_path,
        "--config",    config_path,
    ]

    print("[DEBUG] Zero-123 cmd:", " ".join(map(str, command)))
    subprocess.run(command, check=True)

    print("[INFO] 회전 PNG 저장:", out_path)
    return str(out_path)
