
## 목차

* [1. Stable Diffusion 이란?](#1-stable-diffusion-이란)
* [2. Diffusion Model 과의 차이점](#2-diffusion-model-과의-차이점)
  * [2-1. Latent Space 를 이용](#2-1-latent-space-를-이용)
  * [2-2. Conditioning Mechanism](#2-2-conditioning-mechanism)
* [3. Stable Diffusion 의 구조](#3-stable-diffusion-의-구조)
  * [3-1. Training 시](#3-1-training-시)
  * [3-2. Sampling (= Inference) 시](#3-2-sampling--inference-시)
* [4. Stable Diffusion 사용 사례](#4-stable-diffusion-사용-사례)

## 참고 문헌

* [Amazon AWS, "Stable Diffusion이란 무엇인가요?"](https://aws.amazon.com/ko/what-is/stable-diffusion/)
* [Marvik, "An Introduction to Diffusion Models and Stable Diffusion", 2023.11.28](https://blog.marvik.ai/2023/11/28/an-introduction-to-diffusion-models-and-stable-diffusion/)

## 1. Stable Diffusion 이란?

**Stable Diffusion** 은 [Diffusion Model](Basics_Diffusion%20Model.md) 을 **계산 복잡도 및 효율성** 측면에서 개선한 모델이다.

* 기존의 Diffusion Model 은 **전체 사이즈 이미지를 [U-Net](../Image%20Processing/Model_U-Net.md) 등을 이용하여 처리** 하기 때문에 계산량 및 메모리 사용량이 많다.
* 특히 **Gaussian Noise 로부터 Denoising 하여 이미지를 생성** 하는 과정은 수행 시간이 매우 길다.

## 2. Diffusion Model 과의 차이점

Stable Diffusion 이 기존 Diffusion Model 과 다른 점은 다음과 같다.

| 차이점                                                   | 설명                                                                                                                                                                                      |
|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Latent Space를 이용하는 구조](#2-1-latent-space-를-이용)       | [Auto-Encoder](Basics_Auto%20Encoder.md) 또는 [VAE (Variational Auto-Encoder)](Basics_Variational%20Auto%20Encoder.md) 처럼 Encoder 와 Decoder 를 이용하여 **full-size 이미지를 latent space로 차원 축소** |
| [Conditioning Mechanism](#2-2-conditioning-mechanism) | 특정 Class 의 이미지를 생성하기 위해, **Conditional mechanism (Classifier-Free Guidance, CFG)** 방식 적용                                                                                                |

### 2-1. Latent Space 를 이용

Stable Diffusion 에서는 **계산량 감소를 위해 차원 축소** 를 실시하고, 이를 위해 **Encoder-Decoder 구조를 이용하여 원본 이미지를 Latent Space 로 차원 축소** 한다.

* Encoder-Decoder 구조

![image](images/StableDiffusion_2.PNG)

[(출처)](https://blog.marvik.ai/2023/11/28/an-introduction-to-diffusion-models-and-stable-diffusion/) : Marvik, "An Introduction to Diffusion Models and Stable Diffusion"

* Latent Space 를 이용한 Forward/Reverse Diffusion Process

![image](images/StableDiffusion_1.PNG)

### 2-2. Conditioning Mechanism

## 3. Stable Diffusion 의 구조

### 3-1. Training 시

### 3-2. Sampling (= Inference) 시

## 4. Stable Diffusion 사용 사례

