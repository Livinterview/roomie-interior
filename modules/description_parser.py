import openai, json, re
from pathlib import Path

SYSTEM_PROMPT = """
너는 인테리어 배치 어시스턴트야.
입력 문장과 방 구조 설명을 보고,
가구별 JSON 배열만 반환해줘.
필드: label, rel, color/material, prompt
"""

def parse_description(user_text: str, room_summary: str = "") -> list[dict]:
    messages = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role":"user",   "content": f"[방 구조]\n{room_summary}\n\n[요청]\n{user_text}"}
    ]
    rsp = openai.ChatCompletion.create(
        model = "gpt-4o",
        messages = messages,
        temperature = 0.3
    )
    # JSON 블록만 추출
    match = re.search(r"\\[.*\\]", rsp.choices[0].message.content, re.S)
    data  = json.loads(match.group(0))
    return data
