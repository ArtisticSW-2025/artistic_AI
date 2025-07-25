# artistic_AI

## 피드백 모델

+ 경량화 모델 사용(1B) : https://huggingface.co/qixiangme/hospital-feedback1Bv1

   + CPU 60초 / GPU+양자화 40~50초

+ Blossom (Llama3 8B): https://huggingface.co/qixiangme/hospital-feedback_ver3

    + GPU + 양자화XXXX 

자세한 데이터 형식 및 테스트결과는 노션 이창민 작업현황에서 확인하실수있습니다


feedback.py, feedback_1B.py 둘다 같은 프롬프트 구성 방식을 가집니다

입력해야할 prompt 형식:

1. JSON 형식 -> 프롬프트 String 형식 으로 변환하는 로직입니다
   + 이건 테스트하기 이게 더 편해서 이렇게 한거고 중요한건 프롬프트 String 형식입니다
  
2. 프롬프트 형식은 시스템, 유저 , 어시스턴트로 구성됩니다

```
<|system|>

[system 메시지]

<|user|>

[대화 전문 포함된 user 메시지]

<|assistant|>

[=> 여기에 모델이 피드백 생성]
```

+ 시스템은 AI가 무엇을 해야하는 지 정의를 하는 부분입니다.
    + Training 할때 "너는 병원 예약 도우미 AI가 아닌, '사용자 피드백 코치'야. 아래 대화를 보고 유저가 표현을 개선하거나 주의할 점에 대해 피드백을 작성해줘. 피드백은 간단하게 해주고 능동성, 정확성, 예의 이 3가지 기준으로 점수 매겨줘." 이렇게 Training 했기 때문에 이는 고정시켜주기 바랇니다.
+ User에는 AI에게 줄 입력의 본 부분입니다. 피드백 모델에선 대화 전문을 넣어야합니다. 또한 이 대화 전문의 형식도 Training 데이터와 맞춰야합니다.
    + 처음에 -> "[대화 시작]"
    + 유저가 말한건 "User : [유저가 한 말] "
    + AI가 응답한건 "Assistant: [AI가 한 말] "
    + 대화가 끝날때까지 반복
    + 대화가 끝날때 -> "[대화 끝]"
+ assistant는 assistant가 응답하는 부분입니다.
    + Training 할때는 이 부분에 정답데이터(피드백 해주는 데이터)를 넣어줬지만, 추론(text 생성)할때는 이 부분을 비워줌으로서
    + 파인튜닝 된 AI가 "아 이제 피드백해주는 데이터가 들어갈 차례니 내가 이 빈 부분을 채워줘야겠다" 를 깨닫게 표시해줘야합니다.
 
+ 전체 예시
```
<|system|>
너는 병원 예약 도우미 AI가 아닌, '사용자 피드백 코치'야. 아래 대화를 보고 유저가 표현을 개선하거나 주의할 점에 대해 피드백을 작성해줘. 피드백은 간단하게 해주고 능동성, 정확성, 예의 이 3가지 기준으로 점수 매겨줘.
<|user|>
[대화 시작]
User: 안녕하세요, 수고 많으십니다. 오늘 오전 11시에 진료 예약된 홍길동(010-1234-5678)입니다. 갑작스러운 회사 일로 해당 시간에 방문이 어려워졌습니다. 혹시 오후 2시에서 4시 사이로 시간 조정이 가능할지 여쭙니다.
Assistant: 네, 홍길동 님. 확인해보겠습니다. 아, 마침 2시 30분에 취소 건이 하나 있습니다. 이 시간 괜찮으실까요?
User: 네! 정말 다행이네요. 2시 30분으로 변경 부탁드립니다. 배려해주셔서 정말 감사합니다.
[대화 끝]
<|assistant|>
```

+ ai가 언제 끝내야하는지 잘 모르니 최대 토큰까지 일단 계속 뱉고봅니다. 필요한 것만 파싱하고 쓸모 없는 건 버리는 로직이 필수적입니다




# 대화 생성 모델

GitHub “[DialogueGenModel_finetuning](https://github.com/ArtisticSW-2025/artistic_AI/tree/main/DialogueGenModel_finetuning)” 폴더 안에 다 담아두었습니다.

폴더 안 파일 구성을 간단히 요약하자면 다음과 같습니다.

- [DialogueGenModel_finetuning_medical3.ipynb](https://github.com/ArtisticSW-2025/artistic_AI/blob/main/DialogueGenModel_finetuning/DialogueGenModel_finetuning_medical3.ipynb) : finetuning하고 hugging face에 올리는 과정
- [DialogueGenTest_hugfac.ipynb](https://github.com/ArtisticSW-2025/artistic_AI/blob/main/DialogueGenModel_finetuning/DialogueGenTest_hugfac.ipynb) : hugging face에서 모델 불러와서 대화 생성 하는 과정
- [dialogues_medical3.csv](https://github.com/ArtisticSW-2025/artistic_AI/blob/main/DialogueGenModel_finetuning/dialogues_medical3.csv) : training data CSV 형식 파일
- [dialogues_medical3.json](https://github.com/ArtisticSW-2025/artistic_AI/blob/main/DialogueGenModel_finetuning/dialogues_medical3.json) : training data JSON 형식 파일

파일 안에 전체 과정을 깔끔하게 다 적어두었습니다. 자세한 내용은 직접 파일을 열어서 확인해주세요.

### 참고사항

- training data인 [dialogues_medical3.csv](https://github.com/ArtisticSW-2025/artistic_AI/blob/main/DialogueGenModel_finetuning/dialogues_medical3.csv) 는 일부로 원래보다 훨씬 적은 양을 사용했습니다. finetuning 하는데 너무 오래 걸리기도 하고, 편향된 데이터가 좀 있어서 전처리를 했습니다. 원본 training data는 GitHub [dialogues_medical_fixed(utf-8).csv](https://github.com/ArtisticSW-2025/artistic_AI/blob/main/Finetuning_medical_fixed/dialogues_medical_fixed(utf-8).csv)에 있으니까 참고해주세요.
- inference 과정은 local GPU가 있긴 한데 너무 구려서 Colab GPU로 돌렸습니다.

## 대화 생성 모델 prompt 설명명

### 모델

finetuning 시킬 기본 모델로는 "MLP-KTLim/llama-3-Korean-Bllossom-8B”를 사용했습니다.

finetuning 후 모델 이름 : DialogueGenModel_finetuning_medical3

Hugging Face : https://huggingface.co/wjdbin217/DialogueGenModel_finetuning_medical3

### prompt

대화형 모델에서 prompt는 모델이 대화를 생성하기 위해 참고하는 입력 문장들의 집합입니다.

### prompt 형태

모델이 같기 때문에 위에 창민이형이 써놓은 것이랑 거의 비슷합니다. 

원래 형식은 JSON 형식이지만 파일 내에서는 보기 편하라고 String 형식으로 써놓았습니다.

이 모델에서는 대화 참여자(화자)를 구분하기 위해 role이라는 키워드를 사용합니다.

 “system”, “user”, “assistant”로 구성되어 있으며, 간단한 설명은 다음과 같습니다.

- system : 모델에게 주어지는 역할 또는 지침을 설명
- user : 사용자의 발화
- assistant : 모델의 이전 발화

prompt 예시는 다음과 같습니다.

```python
messages = [      
        {"role": "system", "content": "당신은 경희내과의 병원 콜센터 상담원입니다."
        "모든 응답은 정중하고 친절한 말투로 하며, 병원 콜센터 상담원이 실제 통화에서 말하듯 자연스럽고 공손하게 작성해야 합니다."},
    
    {"role": "user", "content": "안녕하세요. 거기 경희내과 맞나요?"},                                               # user 발화 1
    {"role": "assistant", "content": "네, 맞습니다. 경희내과입니다. 무엇을 도와드릴까요?"},                         # 생성한 assistant 발화 1
    {"role": "user", "content": "아, 네. 예약 변경하려고 하는데요."},                                               # user 발화 2
    {"role": "assistant", "content": "네, 성함 말씀해주시겠어요?"},                                                 # 생성한 assistant 발화 2
    {"role": "user", "content": "네. 홍길동입니다. 오늘 오전 11시 예약이었어요."},                                  # user 발화 3
    {"role": "assistant", "content": "네, 확인되었습니다. 언제로 변경해드릴까요?"},                                 # 생성한 assistant 발화 3
    {"role": "user", "content": "네. 오후 3시로 변경 가능할까요?"},                                                 # user 발화 4
    {"role": "assistant", "content": "네, 가능합니다. 오후 3시로 변경해드릴게요. 다른 궁금한 점 있으신가요?"},      # 생성한 assistant 발화 4
    {"role": "user", "content": "아, 네. 진료 끝나고 진단서 바로 받을 수 있죠?"},                                   # user 발화 5
]
```

이런 구조를 통해 모델은 “이전 대화 흐름”을 이해하고 다음 발화를 자연스럽게 생성할 수 있습니다.

### prompt - system

system은 모델이 수행해야 하는 역할을 정해주는 부분입니다. 쉽게 설명하자면 GPT 쓰는거랑 형태가 비슷합니다. 

이 system 내용이 training data보다 결과에 더 큰 영향을 줍니다. 따라서 구체적으로 작성할 때에는 사소한 내용 까지 조심해야 합니다. 

그래서 처음에는 

```python
messages = [
        {"role": "system", "content": "당신은 경희내과의 병원 콜센터 상담원입니다."
        "사용자는 자신의 이름, 전화번호, 진료 예약 정보를 제공하고, 갑작스러운 일정으로 진료 시간을 변경하고자 전화한 상황입니다."
        "사용자의 진료 시간을 변경해주려면 사용자의 이름과 전화번호를 받아야 합니다.."
        "예약 정보를 확인하고, 변경 가능한 시간대를 안내한 뒤 사용자가 선택한 시간으로 예약을 변경해 주세요. "
        "모든 응답은 정중하고 친절한 말투로 하며, 병원 콜센터 상담원이 실제 통화에서 말하듯 자연스럽고 공손하게 작성해야 합니다."},
```

이렇게 자세하게 쓴 prompt를 사용했다가, 나중에는 다음과 같이 짧고 포괄적인 prompt로 바꿔서 사용했습니다. 

```python
        {"role": "system", "content": "당신은 경희내과의 병원 콜센터 상담원입니다."
        "모든 응답은 정중하고 친절한 말투로 하며, 병원 콜센터 상담원이 실제 통화에서 말하듯 자연스럽고 공손하게 작성해야 합니다."},
```

### prompt - user / assistant

모델에게 줄 input 내용입니다. 이 내용을 모델이 학습하고 그 뒤에 나올 대화를 생성하는 방식입니다.

여기서 user와 assitant로 역할이 구분되어 있는 이유는 양방향 대화 흐름을 모델이 이해할 수 있게 하기 위함입니다. 

좀 tmi 설명을 하자면, 전에 사용했던 kobart 모델은 user만 있는 prompt 였는데요. 일방향 대화 생성 (단발성으로 질문과 응답 한 번씩만) 에는 적합했습니다. 하지만 모의 통화처럼 실시간으로 이전 대화 내용까지는 고려하지 못합니다.

근데 이 prompt는 user와 assistant로 구성이 되어있고, 이전 대화의 흐름을 컨텍스트로 활용하기 때문에, 맥락에 맞는 멀티턴 대화를 생성할 수 있습니다. 이 구조로 되어 있어야 모델이 “누가 말하고 있는지”와 “다음에 누가 말할 차례인지”를 이해할 수 있겠죠?

또 중요한 점은 prompt에서 항상 마지막은 user의 발화여야 한다는 것입니다.

```python
messages = [      
        {"role": "system", "content": "당신은 경희내과의 병원 콜센터 상담원입니다."
        "모든 응답은 정중하고 친절한 말투로 하며, 병원 콜센터 상담원이 실제 통화에서 말하듯 자연스럽고 공손하게 작성해야 합니다."},
    
    {"role": "user", "content": "안녕하세요. 거기 경희내과 맞나요?"},                                               # user 발화 1
    {"role": "assistant", "content": "네, 맞습니다. 경희내과입니다. 무엇을 도와드릴까요?"},                         # 생성한 assistant 발화 1
    {"role": "user", "content": "아, 네. 예약 변경하려고 하는데요."},                                               # user 발화 2
    {"role": "assistant", "content": "네, 성함 말씀해주시겠어요?"},                                                 # 생성한 assistant 발화 2
    {"role": "user", "content": "네. 홍길동입니다. 오늘 오전 11시 예약이었어요."},   
```

이렇게 마지막 발화가 user여야 모델이 “이제 assistant 차례구나”를 알고 대화를 생성할 수 있겠죠?

실제 사용하는 과정은 [DialogueGenTest_hugfac.ipynb](https://github.com/ArtisticSW-2025/artistic_AI/blob/main/DialogueGenModel_finetuning/DialogueGenTest_hugfac.ipynb)에 순서대로 적어두었으니 참고해주세요.

## model
1. 사용자가 대답한 음성을 텍스트로 변환하는 모델 - stt
2. 텍스트를 음성으로 변환 - tts
3. 사용자의 대답을 기반으로 다음 말을 이어나가는 모델 - 대화 생성 model finetuning
4. 사용자가 대답한 음성을 분석해주는 모델 (어조, 말투, 높낮이 등)
5. 전체적인 대화 맥락, 음성 분석 데이터를 보고 종합 평가 하는 모델 (전체적인 평가, 종합 점수, 개선점 등)

## 모의통화 흐름도
1. 상대 AI가 텍스트와 음성을 동시에 출력하며 첫 질문 또는 상황을 제시한다.
2. 사용자가 음성으로 대답한다.
3. STT(Speech-to-Text)를 이용해 음성을 문자로 바꾼다.
4. 변환된 텍스트가 채팅 말풍선 형태로 대화창에 표시된다.
5. 사용자의 발화를 기반으로 AI가 다음 응답을 생성한다.
6. TTS(Text-to-Speech)를 통해 AI의 응답 문장을 음성으로 변환한다.
7. AI의 응답이 텍스트와 음성으로 동시에 출력되고 대화창에 표시된다.
8. 2단계부터 7단계까지의 흐름이 반복된다. (종료 조건이 충족될 때까지 대화가 이어진다.)
9. 대화가 끝나면 전체 발화 내용이 분석된다.
- 말 속도, 어미 사용, 정중함, 끊김 여부 등 다양한 요소가 평가된다.
- 분석 결과를 바탕으로 종합 피드백이 제공된다.
- 분석 점수와 함께 시각적 피드백 및 향후 개선을 위한 맞춤형 연습 가이드가 제시된다.
