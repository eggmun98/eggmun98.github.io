
# 항해플러스 AI 1주차 후기 - 용어 공부의 필요성
소감과 앞으로의 목표
이번 주에 항해플러스 AI 코스의 첫 수업을 들었다. 처음 만난 동기들과 멘토님과의 시간은 생각보다 훨씬 즐거웠다. AI 코스를 통해 얻고자 하는 목표도 명확히 정했다. 이미 제공되는 LLM을 단순히 이용하는 수준에서 벗어나, 내가 직접 무료 오픈소스 LLM을 파인튜닝하여 AI 서비스를 만들 수 있는 수준까지 도달하는 것이다. 이 목표를 이루기 위해 앞으로도 꾸준히 노력할 생각이다.
처음 접한 AI 공부의 어려움
AI 공부를 본격적으로 시작하니, 생소한 용어와 수학 공식들이 쏟아져 나와 처음에는 정신이 없었다. 용어들부터 낯설고 어렵게 느껴졌으며, 수학 공식이 많이 나와 당황스러웠다. 멘토님들은 지금 당장 수학을 깊이 이해하지 않아도 된다고 하셨지만, LLM을 제대로 활용하려면 결국은 깊은 이해가 필요할 것 같다.
그래서 내가 내린 첫 결론은 "용어부터 제대로 이해하자"였다. 용어집을 만들면서 하나하나 의미를 정리하며 공부를 시작했다.

# 중요 용어 쉽게 정리하기
## 머신러닝이란?
컴퓨터가 스스로 문제를 해결하도록 배우는 방법이다.
Task (일): 컴퓨터가 해결해야 할 문제 (예: 사진 속 동물 맞추기)
Evaluation Metric (성적표): 컴퓨터의 성능을 평가하는 지표
Optimization (연습): 컴퓨터가 성능을 높이기 위해 계속 학습하는 과정

## 간단한 머신러닝 모델
Linear Regression (선 긋기): 데이터를 가장 잘 표현하는 직선을 찾는 모델
Logistic Regression (분류 선 긋기): 데이터의 범주를 구분하기 위한 직선을 찾는 모델
리니어 레이어(Linear Layer, 계산기): 입력 숫자에 곱하고 더하여 새로운 숫자를 만드는 층

## 신경망 (딥러닝)
Multi-layer Perceptron (MLP, 여러 층 계산기): 여러 리니어 레이어를 연결하여 복잡한 문제 해결
XOR (특별한 문제): 선 긋기만으로 해결할 수 없어 MLP 같은 신경망이 필요한 문제
ReLU (특별한 버튼): 계산된 값이 마이너스면 0, 플러스면 그대로 사용하는 활성화 함수
Activation Function (활성화 함수): 뉴런의 출력을 결정하는 함수 (예: ReLU, Sigmoid, Tanh 등)

## 딥러닝의 학습 방법
Gradient Descent (내리막길 찾기): 가장 빠르게 정답에 가까워지는 방법
Learning Rate (배우는 속도): 학습할 때 얼마나 빠르게 파라미터를 업데이트할지 결정하는 수치
Parameter (컴퓨터가 기억하는 숫자): 컴퓨터가 학습을 통해 기억하는 숫자
Hyperparameter (컴퓨터가 미리 정한 숫자): 모델의 학습 과정을 제어하는 미리 설정된 숫자들 (예: 학습률, 배치 크기)
Computation Graph (계산 순서 그림): 문제 풀이 과정을 그림으로 나타낸 것
Backpropagation (거꾸로 배우기): 틀린 부분을 뒤에서부터 수정해가는 방법

## 딥러닝 도구와 문제 해결법
PyTorch (컴퓨터 공부 도구): 딥러닝을 쉽게 구현하도록 돕는 프로그램
Tensor (숫자 꾸러미): PyTorch에서 데이터를 나타내는 기본 단위
Stochastic Gradient Descent (조금씩 내려가기): 데이터를 조금씩 보며 빠르게 학습하는 방법
MNIST (숫자 맞추기 게임): 손글씨 숫자를 맞추는 대표적인 데이터셋

## 딥러닝에서 자주 겪는 문제
Overfitting (너무 외우기): 훈련 데이터만 잘 맞추고 새로운 데이터는 못 맞추는 현상
Underfitting (덜 외우기): 모델이 데이터를 제대로 학습하지 못해 성능이 낮은 현상
Generalization Error (새 문제에서 틀리는 것): 새로운 문제에서 발생하는 오차
Validation Data & Early Stopping (미리 확인하고 멈추기): 학습 중간에 성능을 점검하고 더 이상 발전이 없으면 멈추는 방법

## 더 똑똑한 학습 기술들
Weight Decay (숫자 작게 만들기): 불필요한 숫자를 작게 만들어 더 잘 학습하게 하는 방법
Dropout (일부러 잊기): 일부 데이터를 무작위로 제외하여 일반화 성능을 높이는 방법
Advanced Activation Function (더 좋은 버튼): ReLU보다 더 효율적인 다양한 활성화 함수들
Adam Optimizer (똑똑하게 내려가기): 더 빠르고 효율적으로 학습을 진행하는 방법

## 인공지능과 딥러닝의 핵심 용어들
Fine-tuning (미세 조정): 이미 잘 학습된 모델을 특정 데이터에 맞게 추가로 학습시키는 방법
Transfer Learning (지식 옮기기): 다른 일을 하면서 배운 지식을 새로운 문제 해결에 활용하는 방법
Embedding (의미 숫자화): 단어나 문장의 의미를 숫자로 표현한 것
Attention (집중하기): 중요한 정보에 더 집중하여 성능을 높이는 기술
Transformer (주의 깊은 신경망): Attention 기법을 활용해 더 복잡한 데이터를 잘 처리하는 신경망 구조
LLM (Large Language Model, 큰 언어 모델): 엄청난 양의 글을 읽고 자연스러운 문장을 만들어 내는 인공지능 모델 (예: GPT-4)
Prompt (질문 또는 지시): LLM에게 원하는 답을 얻기 위해 입력하는 질문이나 명령어
Context (맥락): LLM이 질문에 답할 때 함께 참고하는 주변 정보
Few-shot Learning (몇 개만 보고 배우기): 데이터를 아주 적게만 봐도 새로운 문제를 해결할 수 있는 능력
Inference (추론, 답하기): 이미 학습된 모델이 새로운 질문에 답하는 과정

# 1주차 과제: MNIST 숫자 분류 모델 만들기
이번 과제는 MNIST라는 손글씨 이미지 데이터를 이용해 0~9까지 숫자를 구별하는 딥러닝 모델을 만드는 것이었다. 이 과제를 통해 회귀와 분류 모델의 차이를 이해하고, 분류 모델 설계 시 중요한 출력층의 크기 설정과 손실 함수 선택에 대해 학습할 수 있었다.
과제의 목표:
1. 분류 문제를 위한 모델 구성과 학습 방법 익히기
2. 모델 정확도 평가 및 과적합 여부 점검하기
과제 수행 과정:
1. MNIST 데이터 로딩 및 전처리
2. 딥러닝 모델 설계 및 구현 (Linear Layers와 ReLU 활용)
3. CrossEntropy 손실함수와 SGD 옵티마이저로 모델 학습
4. 정확도 평가를 통해 성능 확인
약 100회의 Epoch을 진행한 후, 최종 모델의 정확도는 약 94%였다. 이를 통해 기본적인 분류 모델의 학습 방법과 흐름을 명확하게 이해할 수 있었다.

# 마무리하며
항해플러스 AI 1주차 과정을 통해 머신러닝과 딥러닝의 개념과 이론을 명확히 잡을 수 있었다. 가장 기본이 되는 MNIST 문제를 직접 풀어보며 모델 설계부터 학습, 평가까지 전 과정을 경험한 것이 가장 큰 소득이었다. 앞으로 더 심화된 모델을 구축하여 내가 원하는 AI 서비스를 직접 구현할 수 있을 것이라는 자신감을 얻게 된 의미 있는 시간이 되었다.
#항해99 #항해 플러스 AI 후기 #AI 개발자 #LLM


- [테스트](./test.md)