# scripts/download_controlcom_weights.py

from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil
import os

# HuggingFace repo
REPO_ID = "bcmi/ControlCom"

# 다운로드 경로
ckpt_dir = Path("external/ControlCom-Image-Composition/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

print("[*] ControlCom weights 다운로드 시작...")

# 1. ControlCom_view_comp.pth 다운로드
ckpt_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="ControlCom_view_comp.pth",
    cache_dir=str(ckpt_dir),
    local_dir=str(ckpt_dir),
    local_dir_use_symlinks=False
)
print(f"[+] 체크포인트 다운로드 완료: {ckpt_path}")

# 2. openai-clip-vit-large-patch14 폴더 다운로드
clip_files = [
    "openai-clip-vit-large-patch14/config.json",
    "openai-clip-vit-large-patch14/pytorch_model.bin"
]

for file in clip_files:
    out_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=file,
        cache_dir=str(ckpt_dir),
        local_dir=str(ckpt_dir),
        local_dir_use_symlinks=False
    )
    print(f"[+] CLIP 파일 다운로드 완료: {out_path}")

print("[✅] 모든 ControlCom 모델 파일 다운로드 완료!")
