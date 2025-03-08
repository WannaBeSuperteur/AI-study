## 목차

* [1. Image Augmentation 방법 상세](#1-image-augmentation-방법-상세)
  * [1-1. 이미지 형태의 기하학적 변형](#1-1-이미지-형태의-기하학적-변형)
  * [1-2. 이미지 색상 변형](#1-2-이미지-색상-변형)
  * [1-3. 기타 변형 or 노이즈 추가](#1-3-기타-변형-or-노이즈-추가)
* [2. torchvision 을 이용한 Augmentation](#2-torchvision-을-이용한-augmentation)
  * [2-1. 기본 사용법](#2-1-기본-사용법)
  * [2-2. Augmentation 방법 별 함수](#2-2-augmentation-방법-별-함수)
  * [2-3. AutoAugment](#2-3-autoaugment)
* [3. 실험: 최선의 Augmentation 방법 탐색](#3-실험-최선의-augmentation-방법-탐색)
  * [3-1. 실험 설계](#3-1-실험-설계)
  * [3-2. 실험 결과](#3-2-실험-결과)
  * [3-3. 실험 결과에 대한 이유 분석](#3-3-실험-결과에-대한-이유-분석)

## 코드

## 1. Image Augmentation 방법 상세

[Image Augmentation](Basics_Image_Augmentation.md) 의 방법은 다음과 같이 분류할 수 있다.

| 분류              | 상세 유형                                                                                                             | 특징                                                                                                                        |
|-----------------|-------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| 이미지 형태의 기하학적 변형 | - 좌우 반전<br>- 상하 반전<br>- 랜덤한 각도로 회전<br>- Affine 변환<br>- 이미지의 랜덤한 영역만 남기고 잘라내기 (crop)<br>- 이미지의 중심 일정 영역만 남기고 잘라내기  | - Object Detection 과 같은 task 의 경우, bounding box 등 관련 정보도 바뀌어야 함<br>- 물체의 형태가 바뀌면 target value (예: 분류) 가 달라질 수 있는 경우 사용 불가 |
| 이미지 색상 변형       | - 밝기 조정<br>- 대비 (contrast) 조정<br> - 채도 (saturation) 조정<br>- 회색조 (Grayscale)<br>- 반전 (Invert)<br>- 정규화 (Normalize) | - 물체의 색이 바뀌면 target value 가 달라질 수 있는 경우 사용 불가                                                                             |
| 기타 변형 or 노이즈 추가 | - Gaussian Blur<br>- Sharpness 추가<br> - JPEG 이미지로 변환                                                              | - Noise 가 있는 이미지가 많은 데이터셋에 적합할 수 있음                                                                                       |

### 1-1. 이미지 형태의 기하학적 변형

이 방법의 주요 특징은 다음과 같다.

* 물체의 위치가 바뀌므로, 일부 Task 에서 픽셀 좌표 정보가 바뀌어야 함
  * Object Detection 에서의 Bounding Box 의 위치
  * Segmentation 에서의 Segmented pixel 좌표 정보
* 물체의 형태가 바뀌면 Target Value 가 달라지는 경우 사용 불가
  * 예를 들어 MNIST 숫자 분류 데이터셋에서, '6'을 180도 회전하면 '9'가 되어 Class가 달라지는 경우 

![image](images/Image_Augmentation_Methods_2.PNG)

이 방법의 대표적인 예로 다음과 같은 것들이 있다.

* 상하, 좌우 반전
  * 반전 여부만 달라지므로 상하 반전으로 2배, 좌우 반전으로 2배까지 데이터를 증강시킬 수 있다. 
* 랜덤한 각도로 회전
  * 직사각형 모양의 이미지를 회전시킨 후 원래 직사각형에 넣으려고 하면 남는 공간이 생기는데, 이를 적절한 방법으로 채워야 한다. 
* Affine 변환
  * 이미지를 **평행사변형 형태로 변환하거나, 회전하거나, Scaling 하는 등** 다양하게 변형 가능
  * 수학적으로는 다음과 같이 표현된다.

![image](images/Image_Augmentation_Methods_1.PNG)

* 이미지의 랜덤한 영역 / 중심부만 남기고 잘라내기
  * 직사각형 형태로 이미지의 일부분만 남기고 잘라낸다.
  * 물체의 크기 또는 확대 배율이 서로 다른 이미지들이 있는 데이터셋에 적합하다.

### 1-2. 이미지 색상 변형

이 방법의 주요 특징은 다음과 같다.

* 물체의 위치가 바뀌지 않으므로, Object Detection, Segmentation 등의 task 에서도 픽셀 좌표 정보가 바뀌지 않아도 됨
* 물체의 형태가 바뀌면 Target Value 가 달라지는 경우에도 사용 가능
* 물체의 색이 바뀌면 Target Value 가 달라지는 경우 사용 불가
  * 예: 익은 사과와 그렇지 않은 사과를 구분하는 task

![image](images/Image_Augmentation_Methods_3.PNG)

이 방법의 대표적인 예로 다음과 같은 것들이 있다.

* 밝기, 대비 (contrast), 채도 (saturation) 조정
* 회색조 (Grayscale)
* 반전 (Invert)
  * 모든 픽셀에 대해서, R, G, B channel 의 값을 각각 (255 - R), (255 - G), (255 - B) 로 바꾼다. 
* [정규화 (Normalize)](../AI%20Basics/Data%20Science%20Basics/데이터_사이언스_기초_Normalization.md)
  * [Min-max Normalization](../AI%20Basics/Data%20Science%20Basics/데이터_사이언스_기초_Normalization.md#2-1-min-max-normalization) 과 유사하게, 이미지의 픽셀 값을 **-1.0 ~ +1.0 으로 linear 하게** 정규화하는 방법
  * [Gaussian (Z-score) Normalization](../AI%20Basics/Data%20Science%20Basics/데이터_사이언스_기초_Normalization.md#2-2-standarization-z-score-normalization) 과 같이, 이미지의 픽셀 값을 **그 평균, 표준편차를 이용하여 정규화** 하는 방법
    * ImageNet과 같이 유명한 데이터셋은 자체적으로 Gaussian Normalization 을 위한 mean, std-dev 값을 공개하는 경우가 많다. 

### 1-3. 기타 변형 or 노이즈 추가

## 2. torchvision 을 이용한 Augmentation

### 2-1. 기본 사용법

### 2-2. Augmentation 방법 별 함수

### 2-3. AutoAugment

## 3. 실험: 최선의 Augmentation 방법 탐색

### 3-1. 실험 설계

### 3-2. 실험 결과

### 3-3. 실험 결과에 대한 이유 분석