## 목차

* [1. 개요](#1-개요)
* [2. 준비 사항 (GLASS 모델 학습)](#2-준비-사항-glass-모델-학습)
* [3. 실행 방법](#3-실행-방법)

## 1. 개요

* ```glass.py```
  * Vision Anomaly Detection 모델인 GLASS 의 구현 코드
* ```tinyvit.py```
  * Vision Classification 모델인 TinyViT-21M-512-distill 의 구현 코드
  * ```timm``` 라이브러리를 이용하여 간단히 모델을 불러오는 방법으로 [구현할 수 있음](https://huggingface.co/timm/tiny_vit_21m_512.dist_in22k_ft_in1k)
* ```pytorch_grad_cam.py```
  * XAI 모델인 pytorch-grad-cam 의 구현 코드

## 2. 준비 사항 (GLASS 모델 학습)

* 본 디렉토리 ```models``` 의 하위 디렉토리인 ```glass_anomaly_source``` 디렉토리를 다음과 같이 구성
  * [Kaggle Link](https://www.kaggle.com/datasets/jmexpert/describable-textures-dataset-dtd) 에서 이미지 다운로드
  * 위 링크의 ```dtd/images``` 경로 안에 있는 이미지를 다운로드 후, 아래와 같이 복사

```
models
- glass_anomaly_source
  - banded
  - blotchy
  - braided
  ...
  - zigzagged
- glass_original_code
- gradcam_original_code
- common.py
- GLASS.pdf
- glass.py
- pytorch_grad_cam.pdf
- pytorch_grad_cam.py
- README.md
- tinyvit.pdf
- tinyvit.py
```

## 3. 실행 방법

```
python tinyvit.py
python glass.py
python pytorch_grad_cam.py
```