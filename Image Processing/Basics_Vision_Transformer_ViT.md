## 목차

* [1. Vision Transformer (ViT)](#1-vision-transformer-vit)
* [2. ViT 의 구조](#2-vit-의-구조)
  * [2-1. 이미지를 Patch 로 분리](#2-1-이미지를-patch-로-분리) 
  * [2-2. Flattened Patch 에 대한 Linear Projection](#2-2-flattened-patch-에-대한-linear-projection)
  * [2-3. Patch + Position Embedding](#2-3-patch--position-embedding)
  * [2-4. 학습 가능한 Class Embedding](#2-4-학습-가능한-class-embedding)
  * [2-5. Transformer Encoder](#2-5-transformer-encoder)
  * [2-6. MLP Head](#2-6-mlp-head)
* [3. ViT의 응용 모델](#3-vit의-응용-모델)

## 1. Vision Transformer (ViT)

[(논문) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)

**Vision Transformer (ViT)** 는 NLP 에서 자주 쓰이는 [Transformer 모델](../Natural%20Language%20Processing/Basics_트랜스포머%20모델.md) 의 구조를 **언어가 아닌 Vision 문제에 적용** 한 것으로, ICLR 2021에서 위 논문을 통해 발표되었다.

Vision Transformer 의 핵심 아이디어는 다음과 같아.

* 기존의 [CNN (Convolutional Neural Network)](Basics_CNN.md) 의 구조를 Transformer 로 대체
* Transformer 의 계산 효율성이 높은 구조를 이용하므로, 계산 효율성이 높음

Vision Transformer 등장 이후, [PapersWithCode Image Classification task 순위 (ImageNet 기준)](https://paperswithcode.com/sota/image-classification-on-imagenet) 의 순위권에는 ViT 기반 모델이 다수 포진하고 있다.

### 1-1. ViT 모델의 실무적 제약 사항

ViT 기반 모델은 **Transformer 구조의 높은 계산 효율성 + 우수한 성능** 이라는 이점에도 불구하고, **일정한 크기의 Patch 로 이미지를 분할** 한다는 특성 때문에 다음과 같은 실무적 제약 사항이 있다.

* Patch Size 의 배수 해상도가 아닌 이미지는 전처리 없이 사용하려면 resize 해야 한다.
  * 단, 미세한 특징이 중요한 이미지라면 이때 **중요한 정보가 왜곡** 되는 등 문제가 나타날 수 있다. 
* Resize 를 하지 않는 경우, **Padding 을 추가하는 등 전처리** 를 해야 한다.
  * 즉, 전처리를 추가적으로 해야 하기 때문에 **불필요한 연산량이 발생** 한다는 문제가 있다.

## 2. ViT 의 구조

Vision Transformer 의 전체 구조는 다음과 같다.

![image](images/ViT_1.PNG)

[(출처)](https://arxiv.org/pdf/2010.11929) Alexey Dosovitskiy, Lucas Beyer et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021

### 2-1. 이미지를 Patch 로 분리

### 2-2. Flattened Patch 에 대한 Linear Projection

### 2-3. Patch + Position Embedding

### 2-4. 학습 가능한 Class Embedding

### 2-5. Transformer Encoder

### 2-6. MLP Head

## 3. ViT의 응용 모델
