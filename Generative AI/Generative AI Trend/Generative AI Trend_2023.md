# 생성형 AI 트렌드 기술 (2023년)
## Stable Diffusion
Paper : **High-Resolution Image Synthesis with Latent Diffusion Models** (2022) [https://arxiv.org/pdf/2112.10752.pdf](Link)

Citation :
```
title: High-Resolution Image Synthesis with Latent Diffusion Models, 
author: Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer,
year: 2021,
eprint: 2112.10752,
archivePrefix: arXiv,
primaryClass: cs.CV      
```

* Stable Diffusion은 **텍스트를 이미지로 변환하는** 딥러닝 모델이다.
  * Text가 아닌, Layout과 같은 Condition도 입력받을 수 있다.
* Stable Diffusion은 위 논문에서 설명하는 Latent Diffusion 모델의 일종이다.
  * Latent Diffusion 모델은 원본 이미지를 입력값으로 하고 노이즈가 있는 이미지를 출력값으로 하는 모델에 대해, ```원본 이미지 -> Encoder -> Diffusion Process -> Denoising U-net -> Decoder -> 노이즈 이미지``` 와 같이, Diffusion Model과 달리 Encoder와 Decoder가 추가되어 있다.

## GPT-4 Technical Report
Paper : **GPT-4 Technical Report** (2023) [https://arxiv.org/abs/2303.08774.pdf](Link)

Citation :
```
arXiv:2303.08774 [cs.CL]
```

* **ChatGPT에 적용된 GPT-4 모델** 에 대한 논문이다.
  * GPT-4는 GPT-3.5에 비해 Uniform Bar Exam, Leetcode 코딩 테스트 문제 풀이 등에서 좋은 성적을 거두었다.
  * 이미지를 해석하는 등 Vision 기술이 필요한 경우, GPT-4가 GPT-4 (no vision) 에 비해 성능이 좋다.

## Other technologies
* GAN (Generative Adversarial Network)
  * 이미지를 생성하는 Generator와 그 이미지를 구분하는 Discriminator가 경쟁하면서 학습
* AE (Auto-Encoder)
  * 입력 데이터와 출력 데이터가 동일한 형태인 Neural Network
  * 학습된 Neural Network에서 중간의 bottleneck hidden layer를 latent vector (feature) 로 이용하여, 그 값에 따라 이미지 등을 생성
  * Unsupervised Learning을 Supervised Learning 가능한 신경망을 이용하여 해결
* DAE (Denoising Auto-Encoder)
  * 입력 데이터에 random noise 추가
  * noise를 추가하여 Auto-Encoder의 성능에 비해서 향상 시도 (Filter가 Edge를 더 잘 탐지하는 등)
* VAE (Variational Auto-Encoder)
  * Auto-Encoder에서 **MSE 값이 작은 이미지가 의미적으로 먼 경우** 가 있다는 문제점 해결을 위해 등장
  * 샘플링 함수 및 ELBO (Evidence Lower BOund)를 이용