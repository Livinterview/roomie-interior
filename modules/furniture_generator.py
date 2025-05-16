import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

# 1) 배경 제거 모듈 추가 ---------------------- (추가)
from carvekit.api.high import HiInterface

# 2) CarveKit 인터페이스를 전역에서 한 번만 초기화 (추가)
remover = HiInterface(
    object_type="object",  # or "h" for human
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def generate_lora_furniture(prompt: str, obj_id: str) -> str:
    prompt = obj["prompt"]
    base_model = "stabilityai/stable-diffusion-2-1"
    lora_model = "triggah61/lora-home-furniture"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[DEBUG] Loading base model: {base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype
    ).to(device)

    print(f"[DEBUG] Loading LoRA: {lora_model}")
    pipe.load_lora_weights(lora_model)
    print(f"[DEBUG] LoRA 적용 완료")

    print(f"[DEBUG] Running prompt: {prompt}")
    result = pipe(prompt, height=512, width=512)

    if not hasattr(result, "images") or not result.images:
        raise RuntimeError("pipe() 결과에 이미지가 없습니다.")

    image = result.images[0]               # PIL(RGB)
    print("[DEBUG] SD-LoRA 이미지 생성 완료")

    # 3) 배경 제거 -------------------------------- (추가)
    image_rgba = remover([image])[0]       # PIL(RGBA)

    # 4) 저장 -------------------------------------
    save_dir  = Path("assets")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{obj_id}.png"

    print(f"[DEBUG] Saving RGBA to {save_path}")
    image_rgba.save(save_path)             # 투명 PNG
    print(f"[+] 가구 PNG 저장 완료: {save_path}")

    return str(save_path)
