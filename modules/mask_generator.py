# interior/mask_generator.py

from pathlib import Path
import numpy as np
from PIL import Image

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSET_DIR.mkdir(exist_ok=True)

def generate_mask(rotated_img_path: Path, bbox: list[int], object_id: str) -> Path:
    """
    bbox(x, y, w, h)를 기반으로 회전된 가구 이미지용 마스크 생성
    - 255 영역: 가구 삽입 위치
    - 0 영역: 배경 유지
    """
    mask_path = ASSET_DIR / f"mask_{object_id}.jpg"

    # 원본 이미지 크기 가져오기
    original_img = Image.open(rotated_img_path)
    width, height = original_img.size

    # 빈 마스크 생성 (0으로 채움)
    mask = np.zeros((height, width), dtype=np.uint8)

    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x + w, width)
    y2 = min(y + h, height)

    # 해당 영역을 255로 설정
    mask[y1:y2, x1:x2] = 255

    # 저장
    Image.fromarray(mask).save(mask_path)
    print(f"[+] 마스크 생성 완료: {mask_path}")
    return mask_path
