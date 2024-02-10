# GAN (Generative Adversarial Network)

2024.02.10 작성중

## GAN (Generative Adversarial Network) 이란?
**GAN (Generative Adversarial Network, 생성적 적대 신경망)** 이란 unsupervised learning (비지도학습) 의 한 방법으로 생성형 AI에 사용할 수 있는 딥러닝 알고리즘이다.

GAN의 구성 요소는 다음과 같으며, 이들 **생성 모델과 분류 모델이 서로 경쟁** 하는 형태이다.
* **생성 모델** : 이미지를 생성하는 역할을 한다. 분류 모델이 진짜로 판단할 만큼 정교한 가짜 이미지를 생성해서 분류 모델을 속이는 것이 목표이다.
* **분류 모델** : 진짜 이미지 (실제 이미지) 와 생성 모델이 생성한 가짜 이미지를 분류한다.

## GAN의 구조 및 작동 원리

## GAN의 활용 사례

## 다양한 GAN 모델

### DCGAN
**DCGAN (Deep Convolutional Generative Adversarial Network)** 은 GAN의 학습 불안정성을 해결하기 위해 2016년에 구글에서 발표한 딥러닝 모델이다. 이 모델은 다음과 같은 특징을 갖는다.
* 기존의 GAN의 생성 모델 및 분류 모델에는 fully-connected 방식을 적용했는데, 그 대신 convolutional 방식을 채택했다.
  * 이를 통해 보다 향상된 성능과 안정성을 갖는다.
* 생성 모델에서는 factorial strided convolution이라는 방법을 사용하는데, 이것은 입력 사이에 padding, convolution을 적용하여 이미지 크기가 더 커지게 한다.
* 생성 모델이 데이터의 확률분포를 잘 파악하기 때문에 latent vector를 이용하여 결과물을 조작할 수 있다.
  * 이미지 A와 이미지 B의 latent vector의 평균을 계산해서 이미지 A와 이미지 B의 중간 상태에 해당하는 이미지를 생성할 수 있다.
  * 이미지 A와 이미지 B의 차이를 이미지 C에 적용하여 이미지 D를 생성할 때 **이미지를 연산한 것처럼** 결과가 적용된다.
    * 예: (달리는 고양이) - (멈춰 있는 고양이) + (멈춰 있는 토끼) 의 latent vector 를 이용하여 (달리는 토끼) 의 이미지를 생성할 수 있다.