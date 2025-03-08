# 이미지 프로세싱 기초 - Image Augmentation

* [1. Image Augmentation 이란?](#1-image-augmentation-이란)
* [2. Image Augmentation 의 방법](#2-image-augmentation-의-방법)
* [3. Image Augmentation 방법 상세](#3-image-augmentation-방법-상세)

## 1. Image Augmentation 이란?

**Image Augmentation (이미지 증식)** 은 머신러닝 모델을 통해 이미지를 학습시킬 때, 이미지 데이터가 부족한 경우 (특정 class의 이미지가 부족한 경우 포함) 그 개수를 늘리기 위해 사용하는 방법이다.

따라서 다음과 같은 효과가 있다.
* 데이터 양 증가로 인해 overfitting을 줄일 수 있다.
* 이미지 데이터셋 구축에 비용이 많이 드는 (무료 데이터셋은 제한적으로 제공되는 등) 문제를 해결할 수 있다.

## 2. Image Augmentation 의 방법

Image Augmentation의 방법은 다음과 같다.

**이미지의 내용은 유지하고, 틀(?)만 바꾸는 방법**
* 좌우 대칭, 상하 대칭 (Flip)
* ML 모델이 예측하려는 값 (label) 이 바뀌지 않는 범위 내에서의 회전 (예: 좌우 30도 이내로 랜덤하게 회전)
  * 직사각형 모양의 이미지를 90도, 180도, 270도가 아닌 나머지 각도로 회전시켰을 때 나타나는 빈 공간을 채우는 방법 고려 필요
* 확대 및 축소 등 크기 변화

**이미지의 내용을 바꾸는 방법**
* RGB 값, 색상, 채도, 명도를 일정 범위 내에서 변경하는 등 이미지 색을 변환하는 방법
  * Red, Green, Blue 값을 모두 기존의 0.9배 ~ 1.1배 사이의 랜덤한 배수로 만드는 방법
  * 이미지의 색상을 (빨강 <-> 노랑) 의 색상 차이의 최대 30% 만큼 랜덤하게 바꾸는 방법
  * label이 바뀌지 않는 선에서, Green의 값만 기존 값의 특정 범위 내의 랜덤한 배수로 만드는 방법

**주의사항**
* Image Augmentation을 사용하기 위해서는, 그 방법 (예: 회전, 대칭이동) 을 적용해도 머신러닝 모델을 통해 예측하려는 label (예: 특정 분류일 확률, 분류 그 자체 등)이 변화하지 않아야 한다.
  * 예를 들어, MNIST 숫자 (0~9) 분류 문제에서는 특정 숫자를 180도 회전시켜서 다른 숫자로 만들 수 있는 예시가 있으므로 회전은 추천하지 않는다.
* train dataset에만 적용하고, valid, test dataset에서는 적용하지 않아야 한다.

## 3. Image Augmentation 방법 상세

자세한 것은 [해당 문서](Basics_Image_Augmentation_Methods.md) 참고.
