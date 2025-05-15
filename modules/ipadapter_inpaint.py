# interior/ipadapter_inpaint.py

from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSET_DIR.mkdir(exist_ok=True)

pipe = None

def run_ipadapter_inpaint(
    background: Path,
    condition_img: Path,
    mask: Path,
    position: tuple[int, int],
    prompt: str,
    object_id: str
) -> Path:
    """
    IP-Adapter + 마스크 기반 인페인팅 실행
    """
    global pipe

    if pipe is None:
        print("[*] Inpaint 파이프라인 로딩 중...")
        model_id = "stabilityai/stable-diffusion-2-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 이미지 로딩
    image = Image.open(background).convert("RGB").resize((512, 512))
    mask_img = Image.open(mask).convert("L").resize((512, 512))
    condition = Image.open(condition_img).convert("RGB").resize((512, 512))

    # 실제로는 IP-Adapter embedder와 adapter 모델을 끼워 넣는 구조가 필요하지만,
    # 지금은 기본 inpaint로 연결만 해두자 (향후 교체)

    # 기본 인페인팅 실행 (condition을 직접 활용하지 않음 - placeholder)
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_img,
        guidance_scale=7.5,
        num_inference_steps=30,
    ).images[0]

    output_path = ASSET_DIR / f"output_{object_id}.jpg"
    result.save(output_path)
    print(f"[+] Inpainting 결과 저장 완료: {output_path}")
    return output_path
