# Image Processing 프로젝트 3. 흑백 이미지 컬러화 (생성형 AI 적용)
* Dataset: [Mini-ImageNet](https://www.kaggle.com/datasets/deeptrial/miniimagenet/data)
* 수행 기간: 2024.03.20 ~ 2024.03.24 (5일)

## 파일 및 코드 설명
* ```cleanup_data.py``` : 본 프로젝트용 학습 및 테스트 데이터를 다음과 같이 재배치

```
Project 003. 흑백 이미지 컬러화 (생성형 AI 적용)
- archive
  - ImageNet-Mini
    - images
      - n01440764
      - n01443537
      - n01484850
      - ...

에서,

Project 003. 흑백 이미지 컬러화 (생성형 AI 적용)
- images
  - (기존 archive/ImageNet-Mini/images 내부의 모든 폴더에 있는 이미지를 images 폴더에 저장)

로 재배치
```

* ```train.py``` : 학습 과정 전체 진행
  * 출력 모델 : ```main_vae```, ```main_vae_encoder```, ```main_vae_decoder```
* ```test.py``` : 이미지 생성 테스트
  * 필요 모델 : ```main_vae_decoder```
  * 필요 파일 : ```image.png``` (원본 이미지)
  * 출력 파일 : ```output/image_output.png``` (모델에 의해 출력된 이미지)

## 머신러닝 모델 설명
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
|IP-P3|||240320|240324|마스터 브랜치|
|IP-P3-1||```feat```|||데이터 재배치 구현|
|IP-P3-2||```feat```|||모델 학습 부분 구현|
|IP-P3-3||```feat```|||모델 테스트 부분 구현|
|IP-P3-4||```feat```|||모델 성능 향상 시도|