# scripts/wrapper_controlcom.py
import argparse
import subprocess
import json
import tempfile
import sys
from pathlib import Path
import os

# ── 경로 설정 ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CTRL_PATH = ROOT / "external" / "ControlCom-Image-Composition"
TAMING = ROOT / "external" / "taming"

# ── taming 모듈 등록 ─────────────────────
sys.path.insert(0, str(TAMING))

# ── 실행 스크립트 경로 ───────────────────
INFERENCE_SCRIPT = CTRL_PATH / "scripts" / "inference.py"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # 🔧 임시 디렉토리 생성 (ControlCom은 파일 기반 입력을 요구함)
    tmp_dir = Path(tempfile.mkdtemp())
    bg_dir = tmp_dir / "background"
    fg_dir = tmp_dir / "foreground"
    bbox_dir = tmp_dir / "bbox"

    bg_dir.mkdir()
    fg_dir.mkdir()
    bbox_dir.mkdir()

    # 테스트용 더미 파일 생성 (실제 파이프라인에서는 이 부분을 구현해야 함)
    dummy_img = Path("assets/input.jpg")
    dummy_bbox = "256 256 512 512"  # x1, y1, x2, y2 (normalized 아님)

    bg_path = bg_dir / "dummy.jpg"
    fg_path = fg_dir / "dummy.jpg"
    bbox_path = bbox_dir / "dummy.txt"

    bg_path.write_bytes(dummy_img.read_bytes())
    fg_path.write_bytes(dummy_img.read_bytes())
    bbox_path.write_text(dummy_bbox)

    # 🔧 ControlCom inference 실행
    result_dir = tmp_dir / "results"
    result_dir.mkdir()

    cmd = [
        "python", str(INFERENCE_SCRIPT),
        "--testdir", str(tmp_dir),
        "--outdir", str(result_dir),
        "--task", "composition",  # or blending/harmonization/viewsynthesis
        "--skip_grid"
    ]

    print("[DEBUG] subprocess 실행:", " ".join(cmd))

    # subprocess 실행 시 환경변수에 taming 경로 추가
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{str(TAMING)}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, cwd=CTRL_PATH, check=True, env=env)

    # 결과 확인 및 저장
    result_files = list(result_dir.glob("*.jpg")) + list(result_dir.glob("*.png"))
    if not result_files:
        raise RuntimeError("결과 이미지가 생성되지 않았습니다.")

    out = Path(args.out)
    out.write_text(json.dumps({
        "generated_images": [str(p) for p in result_files]
    }, indent=2, ensure_ascii=False))
    print("저장 완료:", out)

if __name__ == "__main__":
    main()
