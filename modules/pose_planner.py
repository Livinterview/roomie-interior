from __future__ import annotations

"""
하이브리드 Pose Planner
========================

1. **LLM** 으로부터 초기 bbox / yaw / pitch 값을 (0~1 정규화 좌표계 기준) 받아옴
2. bbox 값을 `1/GRID_N` 단위 그리드로 스냅(snap)하여 정렬
3. ±`MAX_JITTER` 만큼 좌 · 우 / 상 · 하로 흔들어(jitter) 여러 후보를 만듦
4. 이미 배치된 객체(bbox)와 충돌하는 후보를 제거
5. **FOPA** 점수를 계산해 가장 자연스러운 후보를 선택

가정 및 의존성
--------------
* 배경 이미지가 정사각형이 아닐 수 있으므로, 실제 이미지 크기를 읽어 W/H 를 사용
* `modules.fopa_runner` 안에 `score_bbox(bg_path, obj_dict, bbox_norm) -> float` 함수가 정의되어 있다고 가정
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json, os, re
from PIL import Image  # 배경 이미지 크기 읽기 위해 추가

import openai  # OpenAI Python SDK 사용

# ─────────────────────────── 환경 변수 및 하이퍼파라미터 ───────────────────────────
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 사용할 LLM 모델명
GRID_N = 20          # bbox 스냅용 그리드 분할 수 (1/20 단위)
MAX_JITTER = 0.05    # 후보 좌표를 흔들 때 최대 5% 이동

# ────────────────────────────── LLM 프롬프트 템플릿 ────────────────────────────────
_SYSTEM = """
너는 공간 추론 전문가야. 입력으로 받은 객체를 방 안에 배치해야 해.
아래 형식의 **짧은 JSON** 한 줄만 반환해.
예시: {"x1":0.1,"y1":0.5,"x2":0.35,"y2":0.7,"yaw":90,"pitch":-5}
좌표는 정규화(0~1) 기준, (0,0)은 좌상단, (1,1)은 우하단이야.
"""

_USER = (
    "Place the object '{label}' which should be positioned '{rel}'. "
    "Return bbox and yaw/pitch in JSON (see system instructions)."
)

# ────────────────────────────── 보조 함수들 ──────────────────────────────

def _snap(val: float) -> float:
    """주어진 실수 값을 그리드(1/GRID_N) 단위로 스냅"""
    return round(val * GRID_N) / GRID_N


def _snap_bbox(bbox: List[float]) -> List[float]:
    """bbox 네 점을 모두 스냅"""
    return [_snap(v) for v in bbox]


def _rule_fallback(rel: str) -> Tuple[List[float], float, float]:
    """LLM 장애 시 사용할 간단 휴리스틱 룰"""
    if rel == "left_wall":
        bbox = [0.05, 0.55, 0.25, 0.75];   yaw, pitch =  90, -5
    elif rel == "right_wall":
        bbox = [0.75, 0.55, 0.95, 0.75];   yaw, pitch = -90, -5
    elif rel == "back_wall":
        bbox = [0.35, 0.35, 0.65, 0.55];   yaw, pitch =   0, -5
    else:  # 기타: 중앙 하단
        bbox = [0.30, 0.60, 0.55, 0.80];   yaw, pitch =   0, -5
    return bbox, yaw, pitch

# ────────────────────────────── 메인 함수 ──────────────────────────────

def plan_pose(
    bg_path: Path,
    obj: Dict,
    placed_boxes: Optional[List[List[int]]] = None,
    use_fopa: bool = True,
) -> Tuple[List[int], float, float]:
    """(bbox_px, yaw, pitch)를 반환한다.

    * **bbox_px** : [x1, y1, x2, y2] (배경 해상도 기준 픽셀 좌표)
    * **yaw**, **pitch** : 도(°) 단위 각도
    """
    # LLM을 호출하여 초기 예측값 생성 ------------------------------------------------
    prompt = _USER.format(label=obj.get("label", "object"), rel=obj.get("rel", ""))
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw_json = re.search(r"\{.*\}", resp.choices[0].message.content.strip()).group(0)
        pred = json.loads(raw_json)
        bbox_norm = [pred["x1"], pred["y1"], pred["x2"], pred["y2"]]
        yaw, pitch = float(pred["yaw"]), float(pred["pitch"])
    except Exception:  # LLM 실패 시 룰 기반으로 대체
        bbox_norm, yaw, pitch = _rule_fallback(obj.get("rel", ""))

    # 스냅 및 후보 생성 (속도 최적화)
    # ---------------------------------------------------------
    # 기본적으로는 스냅된 1개 bbox만 사용하고, FOPA를 쓸 때만
    # 4개의 소규모 지터 후보를 추가해 점수를 계산
    bbox_norm = _snap_bbox(bbox_norm)
    candidates = [bbox_norm]

    if use_fopa:
        # 좌/우/상/하 4‑방향으로만 작은 지터(±MAX_JITTER) 생성
        CAND_OFFSET = [(-MAX_JITTER, 0), (MAX_JITTER, 0), (0, -MAX_JITTER), (0, MAX_JITTER)]
        for dx, dy in CAND_OFFSET:
            b = [bbox_norm[0]+dx, bbox_norm[1]+dy, bbox_norm[2]+dx, bbox_norm[3]+dy]
            b = [max(0, min(1, v)) for v in b]
            candidates.append(b)

    # 후보 개수는 최대 5개이므로 계산 비용이 크게 늘지 X

    # 기존 배치와 충돌 제거 ----------------------------------------- -----------------------------------------------------------
    if placed_boxes:
        def overlap(a, b):
            return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])
        candidates = [b for b in candidates if all(not overlap(b, p) for p in placed_boxes)] or [bbox_norm]

    # FOPA 점수로 가장 자연스러운 후보 선택 -------------------------------------------
    if use_fopa:
        try:
            from modules.fopa_runner import score_bbox  # 지연 로딩
            scores = [score_bbox(bg_path, obj, b) for b in candidates]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            bbox_norm = candidates[best_idx]
        except Exception:
            # FOPA 실패 시 첫 후보 사용
            bbox_norm = candidates[0]
    else:
        bbox_norm = candidates[0]

    # 정규화 bbox를 픽셀 좌표로 변환 -----------------------------------------------
    # 배경 이미지 실제 크기 읽기
    with Image.open(bg_path) as im:
        W, H = im.size  # (width, height)
    bbox_px = [
        int(bbox_norm[0] * W), int(bbox_norm[1] * H),
        int(bbox_norm[2] * W), int(bbox_norm[3] * H)
    ]

    return bbox_px, yaw, pitch
