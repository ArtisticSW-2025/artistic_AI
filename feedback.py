import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 피드백 생성 함수, conversation 받아서 prompt 만들고 답변 생성함
def generate_feedback(model, tokenizer, conversation, device='cuda'):
    prompt = ""
    #메시지 받아와서 프롬프트에 넣기 좋게 가공함
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        # 역할별로 토큰 넣어줌. system/user/assistant 나누는 거임
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"

    # 피드백 생성 요청 부분추가, 이제 AI가 입력해야한다는 걸 표시
    prompt += "<|assistant|>\n"

    # 토크나이저로 텍스트 -> 토큰 텐서 변환 후 디바이스에 올림
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids  # 토큰 아이디 

    # 생성 함수 호출 - max 256토큰까지 새 텍스트 생성
    generation_output = model.generate(
        input_ids,
        max_new_tokens=256, # 최대 길이
        do_sample=False,    # 샘플링 안 하고 가장 가능성 높은 토큰만 선택(샘플링하면 가능성 비슷한애들 끼리 랜덤으로 선택함)
        temperature=0.7,    # (샘플링 안 함 근데 남겨둠) 
        top_p=0.9,          # (마찬가지)
        eos_token_id=tokenizer.eos_token_id,  # 생성 종료 토큰
        pad_token_id=tokenizer.pad_token_id   # 패딩 토큰
    )

    # 생성된 텍스트 중 입력 프롬프트 뒷부분부터 디코딩 (원래 입력 제외)
    generated_text = tokenizer.decode(generation_output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return generated_text.strip()


# 유저 발화 부분만 깔끔하게 출력해주는 함수
def print_user_conversation(conversation):
    print("===== 유저 발화 =====")
    for msg in conversation:
        if msg["role"] == "user":
            print(msg["content"].strip())
    print("=====================")


if __name__ == "__main__":
    base_model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"  # 베이스 모델 이름
    adapter_dir = "./results/checkpoint-420"  # 파인튜닝된 adapter 체크포인트 경로
    offload_dir = "./offload"  # GPU 메모리 부족 시 offload할 폴더 지정
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 있으면 GPU로, 없으면 CPU로

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # 패딩 토큰 없으면 eos 토큰으로 설정 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # base 모델 로드 (low_cpu_mem_usage로 메모리 아껴주고 offload 설정도 같이)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map={"": 0},  # 0번 GPU 한개에 올림 서버 하드웨어에 따라 달라져야함 "auto" << 기능도 존재 
        torch_dtype=torch.bfloat16,  
        low_cpu_mem_usage=True,
        offload_folder=offload_dir,
    )

    # 기본모델 + Lora어뎁터 병합
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        offload_folder=offload_dir,
    )
    model.eval()  # 평가모드로 변경, dropout 등 끔

    # 테스트용 대화 예시 (system 역할 + user 역할)
    test_conversation = [
        {
            "role": "system",
            "content": "너는 병원 예약 도우미 AI가 아닌, '사용자 피드백 코치'야. 아래 대화를 보고 유저가 표현을 개선하거나 주의할 점에 대해 피드백을 작성해줘.피드백은 간단하게 해주고 능동성,정확성,예의 이 3가지 기준으로  '총점: xx점' 100점만점으로 반드시 점수를 매겨줘. "
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
