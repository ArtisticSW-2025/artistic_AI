import torch
print(torch.cuda.is_available())  # GPU 사용 가능 여부 출력(True여야 함)
print(torch.cuda.device_count())  # 사용 가능한 GPU 개수 출력 (1 이상이어야 함)
print(torch.cuda.current_device())  # 현재 활성화된 GPU 번호 출력
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # 현재 GPU 이름 출력

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def generate_feedback(model, tokenizer, conversation, device='cuda'):
    """
    모델과 토크나이저, 대화 데이터, 디바이스(GPU/CPU)를 받아
    대화 내용을 토크나이저로 숫자 인코딩 후 모델에 입력하여
    피드백 텍스트를 생성하는 함수.

    conversation : [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    형식의 리스트를 받음.
    """

    prompt = ""
    # 대화 메시지를 역할에 맞춰서 모델이 이해할 수 있는 프롬프트 문자열로 변환
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"

    # 마지막에 어시스턴트가 답변할 차례임을 알리기 위한 토큰 추가
    prompt += "<|assistant|>\n"

    # 프롬프트 텍스트를 토크나이저를 통해 숫자 시퀀스로 변환하고, GPU나 CPU로 텐서 이동
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    # 모델에 입력값을 주고 최대 256토큰까지 새로운 텍스트를 생성
    generation_output = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=False,  # 샘플링 없이 가장 가능성 높은 토큰 선택 (deterministic)
        temperature=0.7,  # 온도 파라미터(샘플링 안 할 때는 무의미)
        top_p=0.9,        # 누적 확률 임계값(샘플링 안 할 때는 무의미)
        eos_token_id=tokenizer.eos_token_id,  # 문장 종료 토큰
        pad_token_id=tokenizer.pad_token_id,  # 패딩 토큰
    )

    # 입력 길이 이후로 생성된 토큰만 디코딩해서 텍스트로 변환
    generated_text = tokenizer.decode(generation_output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return generated_text.strip()


def print_user_conversation(conversation):
    """유저의 발화 부분만 콘솔에 보기 좋게 출력하는 함수"""
    print("===== 유저 발화 =====")
    for msg in conversation:
        if msg["role"] == "user":
            print(msg["content"].strip())
    print("=====================")


if __name__ == "__main__":
    base_model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"  # 베이스 모델 ID (허깅페이스 허브)
    adapter_dir = "qixiangme/hospital-feedback_ver3"       # LoRA 어댑터 디렉토리 or 허깅페이스 허브 경로
    offload_dir = "./offload"  # GPU 메모리 부족 시 일부를 디스크로 오프로딩할 폴더
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 가능하면 cuda, 아니면 cpu

    # 베이스 모델용 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:  # 패딩 토큰이 없으면 eos 토큰으로 대체
        tokenizer.pad_token = tokenizer.eos_token

    # 베이스 모델 로드 (device_map='auto' 로 모델 내부를 GPU/CPU/디스크에 자동 분산 배치)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        offload_folder=offload_dir,  # offload 할 폴더 경로 지정 (필요 시)
    )

    # LoRA 방식으로 파인튜닝된 어댑터 병합
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        offload_folder=offload_dir,
    )

    # model.to(device) 는 offload 사용 시 에러 발생하므로 호출하지 않음

    model.eval()  # 추론 모드로 설정

    # 테스트용 대화 데이터: system 역할에서 AI 역할 정의, user 발화가 포함된 대화 예시
    test_conversation = [
        {
            "role": "system",
            "content": (
                "너는 병원 예약 도우미 AI가 아닌, '사용자 피드백 코치'야. 아래 대화를 보고 유저가 표현을 "
                "개선하거나 주의할 점에 대해 피드백을 작성해줘.피드백은 간단하게 해주고 능동성,정확성,예의 이 3가지 기준으로  "
                "'총점: xx점' 100점만점으로 반드시 점수를 매겨줘. "
            ),
        },
        {
            "role": "user",
            "content": (
                "[대화 시작]\n"
                "User: 예약 좀 바꾸려고 전화했는데요. 오늘 11시 홍길동입니다.\n"
                "Assistant: 네, 홍길동 님. 연락처가 010-1234-5678 맞으신가요?\n"
                "User: 네. 오후에 자리 있나요?\n"
                "Assistant: 네, 3시 15분이 가능합니다.\n"
                "User: 3시 15분이요... 알겠습니다. 그걸로 해주세요. 감사합니다.\n"
                "[대화 끝]"
            ),
        },
    ]

    # 유저 발화만 출력
    print_user_conversation(test_conversation)

    # 모델에 프롬프트 입력해서 피드백 생성
    feedback = generate_feedback(model, tokenizer, test_conversation, device)

    print("\n===== 생성된 피드백 =====")
    print(feedback)
