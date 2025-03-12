## 목차

* [1. LoRA와 QLoRA 요약](#1-lora와-qlora-요약)
* [2. LoRA (Low-Rank Adaptation)](#2-lora-low-rank-adaptation)
  * [2-1. LoRA 의 학습 프로세스](#2-1-lora-의-학습-프로세스)
  * [2-2. LoRA 의 Loss Function](#2-2-lora-의-loss-function) 
  * [2-3. LoRA 의 메모리 사용량 절약 효과](#2-3-lora-의-메모리-사용량-절약-효과)
* [3. QLoRA (Quantized LoRA)](#3-qlora-quantized-lora)
  * [3-1. Paged Optimizer](#3-1-paged-optimizer)

## 1. LoRA와 QLoRA 요약

* [(논문) LoRA: Low-Rank Adaption of Large Language Models](https://arxiv.org/pdf/2106.09685)

[LLM의 Fine Tuning](LLM_기초_Fine_Tuning.md) 방법의 계열 중 [PEFT (Parameter-Efficient Fine-Tuning)](LLM_기초_Fine_Tuning_PEFT.md) 의 일종으로 **LoRA (Low-Rank Adaption)** 와 **QLoRA (Quantized LoRA, Quantized Low-Rank Adaption)** 이 있다.

각 방법에 대해 간단히 정리하면 다음과 같다.

| 방법론   | 설명                                                                                                                                                          |
|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LoRA  | LLM 의 **기존 파라미터 (가중치) 행렬을 저차원 행렬로 분해** 하여 학습<br>- 기존 pre-trained 가중치 행렬에 학습 가능한 Low-rank의 2개의 행렬을 추가하여 학습<br>- 연산량 및 메모리 절약 효과<br>- 최근 가장 많이 쓰이고 있는 PEFT 방법 |
| QLoRA | [양자화 (Quantization)](LLM_기초_Quantization.md) 를 함께 적용한 LoRA 방법                                                                                               |

## 2. LoRA (Low-Rank Adaptation)

**LoRA (Low-Rank Adaptation)** 은 LLM 의 **기존 pre-trained weights 행렬을 저차원으로 분해** 하는 방법이다.
* 실제로는 2개의 레이어 사이에 기존 pre-trained 가중치 (행렬) 가 있을 때, 이들 레이어 간에 **학습 가능한 Low-rank의 2개의 가중치 (행렬) 를 추가** 하는 식으로 구현한다.

![image](images/Fine_Tuning_LoRA_1.PNG)

LoRA 의 특징은 다음과 같다.

* **(장점)** 원래의 High-Rank 행렬 대신 Low-Rank 의 행렬을 2개 사용하기 때문에 **연산량과 메모리가 크게 절약** 된다.
* **(트렌드)** 최근에 가장 많이 쓰이고 있는 PEFT 방법 중 하나이다.

### 2-1. LoRA 의 학습 프로세스

LoRA 의 학습 프로세스는 다음과 같다. 이때, Layer $L1$, $L2$ 사이에 Pre-trained 가중치 행렬 $W_0$ 가 있고, 그 크기는 $p \times q$ 이라 하자.

* 두 레이어 $L1$, $L2$ 사이에 다음의 2개의 가중치 행렬을 위 그림과 같이 연속으로 추가한다.
  * $A$ : 크기는 $p \times r$
  * $B$ : 크기는 $r \times q$
  * 이때 연산 횟수와 메모리를 절약해야 하기 때문에, $r$ 의 값은 $p$, $q$ 보다 일반적으로 매우 작다.
* $A$, $B$ 의 [가중치를 다음과 같이 초기화](../Deep%20Learning%20Basics/딥러닝_기초_Weight_initialization.md) 한다.
  * $A$ : $N(0, \sigma^2)$ 의 정규분포로 Gaussian Initialization
  * $B$ : 모두 0으로 Constant Initialization
* $L1$, $L2$ 사이의 pre-trained 가중치를 비롯한 나머지 부분은 그대로 freeze 시키고, **행렬 $A$, $B$ 만 학습** 시킨다.

이때, $L1$, $L2$ 의 출력값을 각각 $x$, $h$ 라고 한다면, 그 수식은 다음과 같다.

| 구분                 | 수식               |
|--------------------|------------------|
| 기존 Pre-trained LLM | $h = W_0x$       |
| LoRA 적용 시          | $h = W_0x + BAx$ |

### 2-2. LoRA 의 Loss Function

LoRA 의 Loss Function 역시 **next token 예측에 대한 조건부 확률의 곱 (또는 로그 합)** 을 이용할 수 있다. 자세한 것은 [Prefix Tuning 의 Loss Function](LLM_기초_Fine_Tuning_PEFT.md#2-3-prefix-tuning) 부분을 참고.

### 2-3. LoRA 의 메모리 사용량 절약 효과

LoRA 의 메모리 사용량이 행렬의 원소 개수에 비례한다고 가정하고 대략적으로 계산하면 다음과 같다. notation 은 위 2-1 의 내용을 기준으로 한다.

| 구분                         | 메모리 사용량 (행렬의 원소 개수)           |
|----------------------------|-------------------------------|
| 기존 Pre-trained LLM 의 $W_0$ | $p \times q$                  |
| LoRA 적용 가중치 행렬 $A$, $B$    | $(p \times r) + (r \times q)$ |

예를 들어, $p = 4096$, $q = 2048$ 이라고 가정하고 행렬의 원소 개수를 계산하면 다음과 같다.

| 구분       | 기존 $W_0$                              | LoRA 적용 $A$, $B$                                         | 기존 대비 LoRA 의 원소 개수 비율 |
|----------|---------------------------------------|----------------------------------------------------------|-----------------------|
| $r = 16$ | $8.39 \times 10^6 = 4096 \times 2048$ | $9.83 \times 10^4 = (4096 \times 16) + (16 \times 2048)$ | 1.17%                 |
| $r = 64$ | $8.39 \times 10^6 = 4096 \times 2048$ | $3.93 \times 10^5 = (4096 \times 64) + (64 \times 2048)$ | 4.69%                 |

* Single domain 에서의 Fine-tuning 을 위해서는 $r = 16$ 정도면 일반적으로 충분하다.
* 그러나, Multi domain 의 경우 $r = 64$ 정도는 되어야 충분할 수 있다.

## 3. QLoRA (Quantized LoRA)

**QLoRA (Quantized LoRA)** 는 **LoRA 에 [양자화 (Quantization)](LLM_기초_Quantization.md) 를 결합** 한 것이다.

QLoRA 를 적용하는 구체적인 방법은 다음과 같다.

* Pre-trained LLM 부분에는 **[NF4](LLM_기초_Quantization.md#3-양자화-이후의-자료형) 양자화** 적용
* LoRA 를 적용하면서 추가한 Low-Rank 가중치 행렬에는 **BF16 양자화** 적용
* **[이중 양자화 (Double Quantization)](LLM_기초_Quantization.md#4-double-quantization)** 적용
* VRAM 사용량 초과 시 Out Of Memory 오류가 발생하지 않고, 대신 아래의 **Paged Optimization** 기법을 적용한다.

### 3-1. Paged Optimizer

**Paged Optimizer** 는 QLoRA 에서 쓰이는 기술로, **GPU 에서 Out-of Memory (OOM) 오류 발생** 시 [Optimizer](../Deep%20Learning%20Basics/딥러닝_기초_Optimizer.md) 가 자동으로 CPU에서 동작하게 하는 기술이다. 즉 다음과 같이 동작한다.

* Optimizer 의 현재 state가 CPU RAM 으로 이동
* Optimizer 의 갱신을 위해 GPU 에서 필요로 할 때 다시 자동으로 GPU 로 optimizer state 를 이동
