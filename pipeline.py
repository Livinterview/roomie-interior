from pathlib import Path
import uuid

# ── 단계별 모듈 ──────────────────────────────────────────────
from modules.description_parser import parse_description          # 텍스트 → 객체 리스트(JSON)
from modules.pose_planner import plan_pose                        # 객체 → (bbox, yaw, pitch)
from modules.furniture_generator import generate_lora_furniture   # LoRA 가구 PNG
from modules.zero123_runner import rotate_with_zero123            # 회전 뷰 생성
from modules.mask_generator import generate_mask                  # bbox → mask PNG
from modules.fopa_runner import run_fopa_selection                # (선택) 위치 미세 조정
from modules.ipadapter_inpaint import run_ipadapter_inpaint       # IP‑Adapter 기반 인페인팅

# ── 에셋 저장 경로 ───────────────────────────────────────────
ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSET_DIR.mkdir(exist_ok=True)


def run_interior_pipeline(description: str, image_path: Path) -> Path:
    """Roomie Interior 파이프라인 (IP‑Adapter 버전)

    1. LLM / 규칙으로 텍스트 → 객체(JSON)
    2. 객체별로
       a. pose_planner → 초기 (bbox, yaw, pitch)
       b. LoRA로 가구 PNG 생성
       c. Zero123로 yaw/pitch 회전
       d. bbox 기반 mask 생성
       e. (선택) FOPA로 bbox 미세 조정
       f. IP‑Adapter Inpaint로 합성
    3. 최종 합성 이미지 경로 반환
    """

    # Step 1 : 자연어 → 객체 리스트
    objects = parse_description(description)

    # 현재 합성 이미지 (초기 = 배경)
    current_img = image_path

    for obj in objects:
        obj_id = str(uuid.uuid4())[:8]

        # Step 2a. 초기 pose 계산 (bbox, yaw, pitch)
        bbox, yaw, pitch = plan_pose(current_img, obj)

        # Step 2b. LoRA로 정면 가구 이미지 생성
        furniture_png = generate_lora_furniture(obj, obj_id)

        # Step 2c. Zero123로 회전 뷰 생성
        rotated_png = rotate_with_zero123(furniture_png, yaw, pitch, obj_id)

        # Step 2d. bbox → mask 이미지 생성 (흰 = 삽입 영역)
        mask_png = generate_mask(rotated_png, bbox, obj_id)

        # Step 2e. (선택) FOPA로 위치 미세 조정
        best_bbox = run_fopa_selection(current_img, rotated_png, mask_png, bbox) or bbox

        # Step 2f. IP‑Adapter 인페인팅
        current_img = run_ipadapter_inpaint(
            background=current_img,
            condition_img=rotated_png,
            mask=mask_png,
            bbox=best_bbox,
            prompt=obj.get("prompt", ""),
            object_id=obj_id,
        )

    return current_img


if __name__ == "__main__":
    sample_desc = "왼쪽 벽에 민트 소파, 앞에 우든 테이블"
    bg_img = Path("./assets/input.jpg")

    print("[*] 인테리어 파이프라인 실행 시작")
    result = run_interior_pipeline(sample_desc, bg_img)
    print(f"[+] 최종 출력 이미지 경로: {result}")
