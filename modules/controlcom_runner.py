# interior/controlcom_runner.py

def run_controlcom(description: str) -> list[dict]:
    """
    자연어 설명을 파싱해 object별 프롬프트, 위치, 시점 정보 추출
    (임시 dummy 버전: rule 기반 매핑)
    """
    mapping = {
        "소파": {
            "prompt": "a mint green sofa, gray background, studio lighting",
            "bbox": [200, 500, 220, 110],
            "yaw": -60,
            "pitch": 10
        },
        "테이블": {
            "prompt": "a small wooden coffee table, gray background, studio lighting",
            "bbox": [300, 650, 160, 90],
            "yaw": 0,
            "pitch": 15
        }
    }

    parsed_objects = []
    if "소파" in description:
        parsed_objects.append(mapping["소파"])
    if "테이블" in description:
        parsed_objects.append(mapping["테이블"])

    return parsed_objects
