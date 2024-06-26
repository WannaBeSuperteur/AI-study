# Image Processing 프로젝트 3. 흑백 이미지 컬러화 (생성형 AI 적용)
* Dataset: [Flowers](https://www.kaggle.com/datasets/l3llff/flowers)
* 수행 기간: 2024.03.20 ~ 2024.03.29 (10일)

## 파일 및 코드 설명
* ```cleanup_data.py``` : 본 프로젝트용 학습 및 테스트 데이터를 다음과 같이 재배치

```
Project 003. 흑백 이미지 컬러화 (생성형 AI 적용)
- archive
  - flowers
    - astilbe
    - bellflower
    ...

에서, ```archive/flowers/rose``` 디렉토리의 마지막 100장 (test 용도로 사용) 을 제외한 모든 이미지 (899장) 를 112 x 112 로 resize 해서,

Project 003. 흑백 이미지 컬러화 (생성형 AI 적용)
- images
  - (기존 ```archive/flowers/rose``` 디렉토리 내부에 있는 이미지를 images 폴더에 저장)

로 재배치
```

* ```augment_data.py``` : Data Augmentation 실시
  * 필요 파일 : ```images``` 디렉토리 내부의 112 x 112 로 resize 된 이미지 파일
  * 실행 결과 : 1장의 이미지에 대해 crop 된 이미지 4장이 추가되어 기존 데이터 개수 대비 전체 5배의 이미지 데이터 확보
  * **Flowers 데이터셋은 데이터가 충분히 많기 때문에 실행하지 않음**
* ```train.py``` : 학습 과정 전체 진행
  * 출력 모델 : ```main_vae```, ```main_vae_encoder```, ```main_vae_decoder```
  * 참고 : ```train.py``` 실행 시 ```input_convert_test_result``` 디렉토리 및 그 안에 이미지들이 생성되는데, 해당 이미지들은 각 input image 에 대해 **test 시의 결과 이미지를 생성하기 위해 실시하는 변환** 을 그대로 실시한 테스트 이미지로, 실제 모델 학습과 무관 
* ```test.py``` : 이미지 생성 테스트
  * 필요 모델 : ```main_vae_decoder```
  * 필요 파일 : ```test_images``` 폴더 내부의 이미지 파일들 (원본 이미지)
  * 출력 파일 : ```test_output``` 폴더 내부의 이미지 파일들 (모델에 의해 출력된 이미지)
    * ```test_images``` 폴더에 있는 파일들을 crop -> ```112 x 112``` 로 resize 한 후, resize 된 이미지에 대해 테스트 실시 후 결과 파일 출력

## 머신러닝 모델 설명 (VAE 기반 모델)
* **Variational Auto-Encoder 기반의, VAE와 유사한 모델**
* 학습 대상 이미지를 **회색조 (greyscale)** 부분과 **색상 (hue)** 부분으로 각각 분리
  * 색상 부분은 빨간색 (색상 값 ```0```) 을 0도로, 노란색 (색상 값 ```40```) 을 60도로, ..., 색상 값 ```x``` 를 ```1.5x``` 도로 하여 해당 각도의 cos, sin 값을 이용 
* 입력 데이터 : **회색조 부분**
* 출력 데이터 : **색상 및 채도 부분**
  * 입력 데이터로 주어진 회색조 부분에 출력 데이터의 '색상 및 채도 부분'의 color를 입혀서, 색칠된 이미지를 추출
  * cos, sin 값을 각각 y, x 값으로 한 점과 원점을 지나는 직선의 x축의 양의 방향에 대한 각도를 구하고, 해당 각도를 이용하여 색상 값을 구한다.
  * (0, 0) 과 (y, x) 사이의 거리를 이용하여 채도를 구한다.
* 모델 구조
  * Encoder : 입력 데이터 (회색조 부분) 를 입력받아서 latent vector 부분까지 진행
  * Decoder : latent vector 부분에서 시작하여, 입력 데이터 (회색조 부분) 를 또 한번 입력받아서 latent vector와 합성 후, 출력 데이터 (색상 및 채도 부분) 까지 진행

## 실행 순서
```
python cleanup_data.py
python train.py
python test.py
```

## 성능지표 결과
* 성능 측정지표 : 정성 평가로 진행
  * ```test.py``` 파일 실행 시, **얼마나 자연스럽게 color-ize** 되었는지, **생성형 AI의 정의에 부합하도록 이미지의 color가 '생성'되는지** 를 중점적으로 확인

## branch info
|branch|status|type|start|end|description|
|---|---|---|---|---|---|
|IP-P3|||240320|240329|마스터 브랜치|
|IP-P3-1|```done```|```feat```|240320|240320|데이터 재배치 구현|
|IP-P3-2|```done```|```feat```|240320|240320|데이터 augmentation|
|IP-P3-3|```done```|```feat```|240320|240323|모델 학습 부분 구현|
|IP-P3-4|```done```|```feat```|240323|240323|모델 테스트 부분 구현|
|IP-P3-5|```done```|```feat```|240323|240329|모델 성능 향상 시도|