#!/usr/bin/env python3
# infer_zero123.py  (roomie-interior 루트)

import argparse, os, gc, sys, math
from pathlib import Path
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from omegaconf import OmegaConf

# ── CUDA 설정 ────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 레포 경로 등록 ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
ZERO123 = ROOT / "external" / "zero123" / "zero123"
TAMING  = ROOT / "external" / "taming"
sys.path[:0] = [str(ZERO123), str(TAMING)]

from ldm.util import instantiate_from_config          # noqa: E402
from ldm.models.diffusion.ddim import DDIMSampler      # noqa: E402
from geometry.camera import build_camera_tensor        # noqa: E402

# ── 모델 로드 ────────────────────────────────────────────────────
def load_model(cfg_path, ckpt_path):
    gc.collect(); torch.cuda.empty_cache()
    cfg = OmegaConf.load(cfg_path)
    model = instantiate_from_config(cfg.model)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    return model.to(device).eval()

# ── 실행 루틴 ────────────────────────────────────────────────────
def main(a):
    model    = load_model(a.config, a.checkpoint)
    sampler  = DDIMSampler(model)

    # 1. RGB → latent (4×32×32)
    img = Image.open(a.input).convert("RGB").resize((a.size, a.size))
    x   = to_tensor(img).unsqueeze(0).to(device) * 2.0 - 1.0
    z   = model.get_first_stage_encoding(model.encode_first_stage(x))

    # 2. ray map (4-채널: dx,dy,dz,1) — yaw/pitch 반영
    ray_map = build_camera_tensor(
        yaw   = a.yaw,
        pitch = a.pitch,
        roll  = 0.0,
        fov   = 60.0,
        H = 32, W = 32,
        device = device)                    # [1,4,32,32]

    # 3. dummy cross-attn (shape 충족용)
    txt = model.get_learned_conditioning([""])          # [1,77,768]

    cond = {"c_concat": [ray_map], "c_crossattn": [txt]}

    # 4. DDIM 1-step 샘플링
    with torch.no_grad():
        sample, _ = sampler.sample(
            S           = 1,
            conditioning= cond,
            batch_size  = 1,
            shape       = [4, 32, 32],
            verbose     = False,
            x_T         = z          # start from encoded latent
        )
        rgb = model.decode_first_stage(sample)          # [-1,1]
        rgb = (rgb.clamp(-1, 1) + 1) / 2               # [0,1]

    # 5. 저장
    out = (rgb * 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_path = a.output if a.output.endswith(".png") else a.output + ".png"
    Image.fromarray(out).save(out_path)
    print("✅ 저장 완료:", out_path)

# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--yaw",  type=float, required=True)
    p.add_argument("--pitch",type=float, required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--size", type=int, default=256)
    main(p.parse_args())
