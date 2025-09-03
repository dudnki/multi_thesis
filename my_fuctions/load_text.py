import json

def default_prompts_from_json(classes, json_path):
    """
    JSON 파일에서 커스텀 프롬프트를 불러오고,
    없는 클래스는 제네릭 프롬프트로 생성합니다.
    """
    # 1. JSON 파일을 열고 커스텀 프롬프트 데이터를 불러옵니다.
    with open(json_path, 'r', encoding='utf-8') as f:
        custom_prompts = json.load(f)

    # 2. 커스텀 프롬프트가 정의된 클래스 목록을 자동으로 생성합니다.
    in_text_list = list(custom_prompts.keys())

    # 3. 제네릭 프롬프트를 위한 템플릿을 정의합니다.
    tmpl = [
        "{c} has distinctive coat patterns.",
        "{c} shows a characteristic ear shape.",
        "{c} typically has specific eye shape and color.",
        "{c} has a typical body size and proportions.",
        "{c} has a recognizable muzzle and face structure.",
        "{c} has a notable tail shape and carriage.",
        "{c} coat length and texture help identify it.",
    ]

    # 4. 최종 프롬프트 딕셔너리를 생성합니다.
    d = {}
    for c in classes:
        if c in custom_prompts:
            # JSON에 있는 클래스는 커스텀 프롬프트를 사용합니다.
            d[c] = custom_prompts[c]
        else:
            # 없는 클래스는 기존 템플릿으로 생성합니다.
            d[c] = [t.format(c=c) for t in tmpl]
    return d, in_text_list