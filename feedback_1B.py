
# CPU 사용
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def generate_feedback(model, tokenizer, conversation, device='cpu'):
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

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    with torch.no_grad():
        generation_output = model.generate(
            input_ids,
            max_new_tokens=300,
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
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_dir = "qixiangme/hospital-feedback1Bv1"
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (CPU + float32)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        torch_dtype=torch.float32
    )

    model.eval()

    test_conversation = [
        {
            "role": "system",
            "content": "너는 병원 예약 도우미 AI가 아닌, '사용자 피드백 코치'야. 아래 대화를 보고 유저가 표현을 개선하거나 주의할 점에 대해 피드백을 작성해줘. 피드백은 간단하게 해주고 능동성, 정확성, 예의 이 3가지 기준으로 점수 매겨줘."
        },
        {
            "role": "user",
            "content": "[대화 시작]\nUser: 안녕하세요, 수고 많으십니다. 오늘 오전 11시에 진료 예약된 홍길동(010-1234-5678)입니다. 갑작스러운 회사 일로 해당 시간에 방문이 어려워졌습니다. 혹시 오후 2시에서 4시 사이로 시간 조정이 가능할지 여쭙니다.\nAssistant: 네, 홍길동 님. 확인해보겠습니다. 아, 마침 2시 30분에 취소 건이 하나 있습니다. 이 시간 괜찮으실까요?\nUser: 네! 정말 다행이네요. 2시 30분으로 변경 부탁드립니다. 배려해주셔서 정말 감사합니다.\n[대화 끝]"
        },
    ]

    print_user_conversation(test_conversation)
    feedback = generate_feedback(model, tokenizer, test_conversation, device)
    print("\n===== 생성된 피드백 =====")
    print(feedback)




#GPU + 양자화 사용
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

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

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=300,
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
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_dir = "qixiangme/hospital-feedback1Bv1"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with 4bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
    )

    model.eval()

    test_conversation = [
        {
            "role": "system",
            "content": "너는 병원 예약 도우미 AI가 아닌, '사용자 피드백 코치'야. 아래 대화를 보고 유저가 표현을 개선하거나 주의할 점에 대해 피드백을 작성해줘. 피드백은 간단하게 해주고 능동성, 정확성, 예의 이 3가지 기준으로 점수 매겨줘."
        },
        {
            "role": "user",
            "content": "[대화 시작]\nUser: 안녕하세요, 수고 많으십니다. 오늘 오전 11시에 진료 예약된 홍길동(010-1234-5678)입니다. 갑작스러운 회사 일로 해당 시간에 방문이 어려워졌습니다. 혹시 오후 2시에서 4시 사이로 시간 조정이 가능할지 여쭙니다.\nAssistant: 네, 홍길동 님. 확인해보겠습니다. 아, 마침 2시 30분에 취소 건이 하나 있습니다. 이 시간 괜찮으실까요?\nUser: 네! 정말 다행이네요. 2시 30분으로 변경 부탁드립니다. 배려해주셔서 정말 감사합니다.\n[대화 끝]"
        },
    ]

    print_user_conversation(test_conversation)
    feedback = generate_feedback(model, tokenizer, test_conversation)
    print("\n===== 생성된 피드백 =====")
    print(feedback)
