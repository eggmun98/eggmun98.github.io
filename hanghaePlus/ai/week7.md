## 항해 플러스 7주차 후기 - Wandb와 Logger로 GPT 학습 효율 높이기

항해 플러스 AI 7주차는 본격적으로 HuggingFace를 활용한 LLM Pre-training과 Fine-tuning을 제대로 경험하고, Python Logging과 Wandb라는 두 가지 강력한 도구까지 배우면서 그야말로 AI 개발자로서의 역량을 한층 높일 수 있었던 알찬 시간이었습니다.

이번 글에서는 7주차에 배운 핵심 내용과, GPT 모델 Fine-tuning 실습 과정에서 배운 값진 경험들을 생생하게 전달하고자 합니다.

### 📌 Python Logger: print()의 시대는 갔다!

이전까지 저는 모델의 학습 상태를 확인할 때면 습관적으로 print()를 남발했어요. 하지만 모델 학습이 복잡해질수록 터미널에서 로그가 뒤엉키는 현상이 잦았죠. 이번 주 Python Logger를 배우고 나서는 이 모든 문제가 마법처럼 사라졌답니다.

logging 라이브러리는 로그 레벨을 조절해 원하는 정보만 선택적으로 확인할 수 있고, 로그를 파일에 깔끔하게 저장할 수도 있어서 너무 편했어요. 특히 로그 포맷을 설정할 수 있어, 가독성을 높이니 모델의 상태를 빠르게 파악하는 데 정말 큰 도움이 됐습니다.

예를 들어, 학습 초기에는 디버깅을 위해 DEBUG로 세부 사항을 모두 찍다가, 안정화 단계에서는 INFO로 설정을 바꾸면 깔끔하게 요점만 남길 수 있었어요. 작은 변화였지만 생산성에 엄청난 차이를 만들었습니다.

### 📌 Wandb: 모델의 심장 소리를 실시간으로 듣다

Logger가 일종의 ‘진단 장비’라면, Wandb(Weights & Biases)는 제 모델의 생생한 심장 소리를 들려주는 ‘모니터링 시스템’ 같았습니다. 특히 Wandb는 단순히 숫자를 출력하는 것을 넘어 실험 결과를 실시간으로 웹에서 시각화하고 공유할 수 있게 해주는 서비스였어요.

Wandb의 장점 중 가장 인상적이었던 건:

실험 모니터링: 실시간으로 loss를 그래프로 확인하며 모델의 건강 상태를 빠르게 체크할 수 있었습니다.

데이터 시각화: GPU 사용률, grad_norm 등 다양한 지표를 한 화면에서 깔끔하게 볼 수 있어 직관적이었습니다.

협업의 용이성: 팀원들과 Wandb 링크만 공유하면, 원격으로 실험 상태를 함께 보며 빠르게 피드백을 받을 수 있었습니다.

덕분에 학습 도중 GPU 메모리가 터지거나 loss가 폭주하는 상황을 즉시 발견하고 조치할 수 있었어요. Wandb 덕분에 개발의 품질이 확 올라가는 느낌을 강하게 받았습니다!

### 📌 GPT Fine-tuning 실습: Validation으로 완성도를 높이다
![train vs eval loss](img/week7-trainVsEvalLoss.png)

이번 주차의 메인 과제는 HuggingFace Trainer를 활용한 GPT 모델 Fine-tuning이었어요. HuggingFace를 이용하면 데이터셋 로딩, 토큰화, 모델 설정까지 간단하게 처리할 수 있어서 이전보다 훨씬 편했습니다.

특히 이번에는 validation 데이터를 추가하여 평가 단계에서 과적합 여부까지 확인하는 부분이 가장 인상적이었어요. 기존에 과적합을 단순히 "직감"으로만 판단했다면, 이번에는 확실히 수치적으로 파악할 수 있었죠.

데이터셋: wikitext-2-raw-v1 (95% train, 5% valid)

모델: GPT (openai-community/openai-gpt)

평가 주기: logging_steps=100, eval_steps=500

학습을 진행하면서 Wandb를 통해 train/loss와 eval/loss를 동시에 확인하니, 처음엔 같이 내려가다가 일정 시점부터 eval loss가 높아지는 현상을 볼 수 있었어요. 이때 바로 학습을 멈추고 early stopping을 적용해 성능 저하를 미연에 방지했습니다. Wandb 덕분에 시각적으로 빠르게 대응할 수 있어, validation 데이터의 중요성을 온몸으로 느낄 수 있었어요.

### 📌 삽질의 교훈: Precision과 GPU 메모리의 치열한 싸움

모델 학습 과정 중 가장 기억에 남는 삽질은 torch_dtype 선택 문제였어요. 처음에는 메모리를 아끼려고 float16으로 설정했는데, 학습 중 gradient가 터지면서 loss가 nan이 되는 바람에 많이 고생했죠. 결국, bfloat16을 사용하고 gradient accumulation을 적용해 메모리와 안정성을 동시에 잡았어요.

이 경험을 통해 Precision 설정이 단순히 성능뿐 아니라 안정성까지 좌우할 수 있다는 중요한 교훈을 얻었습니다.

### 📌 LLM Pre-training부터 SFT, DPO까지: LLM의 성장 과정 완벽 정리

이번 주는 이론 학습도 풍성했어요. LLM의 전반적인 학습 과정인 Pre-training, Instruction-Tuning(SFT), 그리고 최근 주목받고 있는 DPO(Direct Preference Optimization)까지 상세히 배웠습니다.

특히 Preference Data를 사용해 LLM의 답변을 더 정교하고 사람이 원하는 형태로 align하는 RLHF, DPO 같은 기법을 통해 LLM이 어떻게 발전하고 있는지를 이해할 수 있었어요. DPO는 별도의 Reward Model 없이 Preference 데이터만으로 한 단계 만에 alignment를 완성할 수 있다는 점이 흥미로웠습니다.

### 📌 마무리하며

이번 7주차는 모델을 학습시키는 과정에서 놓치기 쉬운 로그 관리와 모니터링을 체계적으로 배울 수 있었던 귀중한 시간이었습니다. Python Logger로 효율적으로 학습 로그를 관리하고, Wandb를 통해 학습 과정을 직관적으로 모니터링하면서, 이전보다 훨씬 안정적이고 체계적인 실험 환경을 구축할 수 있었어요.

특히, HuggingFace Trainer로 GPT 모델을 Fine-tuning하며 validation 데이터를 활용하여 성능을 평가하는 과정을 통해 모델이 더 이상 숫자로만 존재하지 않고, 학습을 통해 실제로 발전해 나가는 생생한 과정을 직접 느낄 수 있었습니다. 앞으로도 이런 디테일한 부분까지 놓치지 않고 챙기며, 더욱 견고하고 섬세한 AI 개발자로 성장하겠습니다.

항해 플러스 — 추천인 코드: CF7LUQ

#항해99 #항해플러스AI후기 #AI개발자 #LLM #Wandb #HuggingFace #GPT