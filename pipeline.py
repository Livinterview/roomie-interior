# interior/pipeline.py

from pathlib import Path
import uuid

from modules.controlcom_runner import run_controlcom
from modules.furniture_generator import generate_lora_furniture
from modules.zero123_runner import rotate_with_zero123
from modules.mask_generator import generate_mask
from modules.fopa_runner import run_fopa_selection
from modules.ipadapter_inpaint import run_ipadapter_inpaint

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSET_DIR.mkdir(exist_ok=True)

def run_interior_pipeline(description: str, image_path: Path) -> Path:
    """
    전체 인테리어 파이프라인 실행
    1. 설명 파싱 → ControlCom
    2. 객체별:
        a. LoRA로 가구 생성
        b. Zero123으로 회전
        c. 마스크 생성
        d. FOPA로 최적 위치 선정
        e. IP-Adapter로 삽입
    3. 최종 이미지 반환
    """
    # Step 1. ControlCom으로 자연어 → object 배치 정보 추출
    objects = run_controlcom(description)
    current_img = image_path

    for obj in objects:
        obj_id = str(uuid.uuid4())[:8]
        prompt = obj["prompt"]
        yaw, pitch = obj["yaw"], obj["pitch"]
        bbox = obj["bbox"]  # 예: [x, y, w, h]

        # Step 2a. LoRA로 정면 가구 이미지 생성
        object_img = generate_lora_furniture(prompt, obj_id)

        # Step 2b. Zero123으로 회전
        rotated_img = rotate_with_zero123(object_img, yaw, pitch, obj_id)

        # Step 2c. 마스크 생성
        mask = generate_mask(rotated_img, bbox, obj_id)

        # Step 2d. FOPA로 위치 최적화 (선택적)
        best_pos = run_fopa_selection(current_img, rotated_img, mask, bbox)

        # Step 2e. 인페인팅
        current_img = run_ipadapter_inpaint(
            background=current_img,
            condition_img=rotated_img,
            mask=mask,
            position=best_pos,
            prompt=prompt,
            object_id=obj_id
        )

    return current_img

if __name__ == "__main__":
    from pathlib import Path

    description = "왼쪽 벽에 민트 소파, 앞에 테이블"
    image_path = Path("./assets/input.jpg")  # input 이미지 미리 넣어둘 것

    print("[*] 인테리어 파이프라인 실행 시작")
    output = run_interior_pipeline(description, image_path)
    print(f"[+] 최종 출력 이미지 경로: {output}")