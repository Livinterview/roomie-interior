import torch
from diffusers import StableDiffusionPipeline

def generate_lora_furniture(prompt, obj_id):
    base_model = "stabilityai/stable-diffusion-2-1"
    lora_model = "triggah61/lora-home-furniture"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    pipe.load_lora_weights(lora_model)

    image = pipe(prompt, height=512, width=512).images[0]

    save_path = f"interior/assets/{obj_id}.png"
    image.save(save_path)
    print(f"[+] 가구 이미지 생성 완료: {save_path}")
    return save_path
