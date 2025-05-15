# interior/fopa_runner.py

from pathlib import Path
import shutil
import subprocess
import json
import numpy as np
from PIL import Image

FOPA_DIR = Path(__file__).resolve().parent.parent / "fopa"
FOPA_DATA_DIR = FOPA_DIR / "data/data"
ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"

def write_fopa_test_json(ann_id, sc_id, new_w, new_h, scale, pos_label):
    test_json = [{
        "id": "test1",
        "scID": sc_id,
        "annID": ann_id,
        "bg": f"{sc_id}.jpg",
        "fg": f"{ann_id}.jpg",
        "mask": "",
        "fg_class": "object",
        "composite_fg": f"{ann_id}.jpg",
        "newWidth": new_w,
        "newHeight": new_h,
        "scale": scale,
        "pos_label": [pos_label],
        "neg_label": [[50, 50]]
    }]
    with open(FOPA_DATA_DIR / "test_pair_new.json", "w") as f:
        json.dump(test_json, f, indent=2)

def get_best_position_from_heatmap(ann_id, new_w, new_h, scale):
    key = f"{ann_id}_1_{new_w}_{new_h}_{scale}"
    heatmap_path = FOPA_DIR / "best_weight_test_heatmap" / f"{key}.jpg"

    if not heatmap_path.exists():
        print(f"[-] Heatmap 없음: {heatmap_path}")
        return None

    arr = np.array(Image.open(heatmap_path).convert("L"))
    y, x = np.unravel_index(np.argmax(arr), arr.shape)
    return (x, y)

def run_fopa_selection(bg_path: Path, fg_path: Path, mask_path: Path, bbox: list[int]) -> tuple[int, int]:
    """
    FOPA를 실행해서 가장 자연스러운 배치 위치(x, y)를 반환
    """
    ann_id = "3"  # 객체 ID 고정 또는 uuid도 가능
    sc_id = "000000000001"
    new_w, new_h = 200, 100  # 크기 고정 or bbox 비율 기반 추후 개선
    scale = 0.6
    pos_label = [bbox[0], bbox[1]]  # 최초 위치 기준

    # FOPA 입력 경로 세팅
    shutil.copy(bg_path, FOPA_DATA_DIR / "bg" / f"{sc_id}.jpg")
    shutil.copy(fg_path, FOPA_DATA_DIR / "fg/foreground" / f"{ann_id}.jpg")
    shutil.copy(mask_path, FOPA_DATA_DIR / "fg/test" / f"{ann_id}_1_200_100.jpg")
    shutil.copy(mask_path, FOPA_DATA_DIR / "mask/test" / f"{ann_id}_1_200_100.jpg")

    write_fopa_test_json(ann_id, sc_id, new_w, new_h, scale, pos_label)

    # FOPA 실행
    subprocess.run(["python", "test.py", "--mode", "heatmap"], cwd=FOPA_DIR)
    subprocess.run(["python", "test.py", "--mode", "composite"], cwd=FOPA_DIR)

    best_pos = get_best_position_from_heatmap(ann_id, new_w, new_h, scale)
    print(f"[+] FOPA 최적 위치: {best_pos}")
    return best_pos if best_pos else (bbox[0], bbox[1])  # fallback
