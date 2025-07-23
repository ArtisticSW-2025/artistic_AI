import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

#모델, 토크나이저 , 데이터, GPU(device) 필요
def generate_feedback(model, tokenizer, conversation, device='cuda'):
    prompt = ""
    #data.json에서 가져와서, json형식을 -> string으로 바꿔, 프롬프트에 넣을준비
    #AI에게 줘야할 프롬프트의 형식 : System(AI의 역할 정의) , User(프롬프트 값),하고 마지막에 <Assistant>추가해줘서 이제 AI가 아 내가 입력할 차례구나 인식하게
    #AI에게 줄 프롬프트 예시
    """<|system|>
        너는 병원 예약 도우미 AI가 아닌, '사용자 피드백 코치'야. 아래 대화를 보고 유저가 표현을 개선하거나 주의할 점에 대해 피드백을 작성해줘. 피드백은 간단하게 해주고 능동성, 정확성, 예의 이 3가지 기준으로 '총점: xx점' 100점만점으로 반드시 점수를 매겨줘.
        <|user|>
        [대화 시작]
        User: 예약 좀 바꾸려고 전화했는데요. 오늘 11시 홍길동입니다.
        Assistant: 네, 홍길동 님. 연락처가 010-1234-5678 맞으신가요?
        User: 네. 오후에 자리 있나요?
        Assistant: 네, 3시 15분이 가능합니다.
        User: 3시 15분이요... 알겠습니다. 그걸로 해주세요. 감사합니다.
        [대화 끝]
        <|assistant|>"""
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"

    prompt += "<|assistant|>\n"
    #데이터를 숫자로 바꿔야하므로, tokenizer을 써서 text -> 숫자 정보로(Python Torch), device(GPU)에 할당 명시
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    input_ids = inputs.input_ids

    #이제 model이 text generate, input_ids 줘서 input 명시
    generation_output = model.generate(
        input_ids, #input 명시
        max_new_tokens=256, #최대 길이 지정
        do_sample=False, #samlple이라고 text generate할때 쓰는 방식이 있는데 일단 이거 False(너무 랜덤이라서)
        eos_token_id=tokenizer.eos_token_id, #EndOfSystem 줄임말, 언제 끝나는 지 명시
        pad_token_id=tokenizer.pad_token_id,  # 패딩 토큰 명시
    )
    # 기존 입력 이후로 생성된 부분만 디코딩
    generated_text = tokenizer.decode(generation_output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return generated_text.strip()


def print_user_conversation(conversation):
    print("===== 유저 발화 =====")
    for msg in conversation:
        if msg["role"] == "user":
            print(msg["content"].strip())
    print("=====================")


if __name__ == "__main__":
    #학습시킨게 Lora방식이라 BaseModel과 내가만든 adapter 두게 로딩 필요
    base_model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    adapter_dir = "./results//checkpoint-420"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #토크나이저 base_model에서 불러옴
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #사전 학습된 Base 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    #Base Model과 LoRa Adapter 병합
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # 모델을 GPU 또는 CPU로 이동
    model.to(device)
    # 추론 모드로 설정 
    model.eval()
    
    test_conversation = [
        {
            "role": "system",
            "content":"너는 병원 예약 도우미 AI가 아닌, '사용자 피드백 코치'야. 아래 대화를 보고 유저가 표현을 개선하거나 주의할 점에 대해 피드백을 작성해줘.피드백은 간단하게 해주고 능동성,정확성,예의 이 3가지 기준으로  '총점: xx점' 100점만점으로 반드시 점수를 매겨줘. "
        },
        {
            "role": "user",
            "content": "[대화 시작]\nUser: 예약 좀 바꾸려고 전화했는데요. 오늘 11시 홍길동입니다.\nAssistant: 네, 홍길동 님. 연락처가 010-1234-5678 맞으신가요?\nUser: 네. 오후에 자리 있나요?\nAssistant: 네, 3시 15분이 가능합니다.\nUser: 3시 15분이요... 알겠습니다. 그걸로 해주세요. 감사합니다.\n[대화 끝]"
        },
    ]

    print_user_conversation(test_conversation)
    feedback = generate_feedback(model, tokenizer, test_conversation, device)
    print("\n===== 생성된 피드백 =====")
    print(feedback)
