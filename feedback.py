import torch
print(torch.cuda.is_available())  # True여야 함
print(torch.cuda.device_count())  # 1 이상이어야 함
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def generate_feedback(model, tokenizer, conversation, device='cuda'):
    prompt = ""
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

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    generation_output = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_text = tokenizer.decode(generation_output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return generated_text.strip()


def print_user_conversation(conversation):
    print("===== 유저 발화 =====")
    for msg in conversation:
        if msg["role"] == "user":
            print(msg["content"].strip())
    print("=====================")


if __name__ == "__main__":
    base_model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    adapter_dir = "qixiangme/hospital-feedback_ver3"
    offload_dir = "./offload"  # GPU 부족 대비
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        offload_folder=offload_dir,  #
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        offload_folder=offload_dir,
        
    )
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
