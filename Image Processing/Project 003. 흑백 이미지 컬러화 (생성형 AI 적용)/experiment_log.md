# Model 3 : skip connection 추가 적용 및 MSE Loss 가중치 4.0 적용 (2024.03.27 22시)
* 처음 **100장** 의 이미지만 이용하여 학습
* epoch **5회**
* MSE Loss 가중치 : **4.0**
* 일부 이미지에 대해 빨간색으로 칠해진 부분 근처에서 색상 그라데이션이 어색한 부분은 딥러닝 모델이 아닌 **test image 생성 알고리즘** 이슈이며 현재 해결한 상태임.

## 직전 모델 (Model 2) 대비 변동 사항
**2024.03.27 21시** (skip connection 추가 적용) 에서, MSE Loss 가중치를 2.5 에서 4.0 으로 변경

## 학습 로그
* 최종 Loss : **0.1190** (Model 1 대비 **27.2%** 감소)

```
2024-03-27 21:54:51.393980: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-03-27 21:54:51.917934: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
19584/19584 [==============================] - 166s 8ms/sample - loss: 0.4333
Epoch 2/5
19584/19584 [==============================] - 150s 8ms/sample - loss: 0.1650
Epoch 3/5
19584/19584 [==============================] - 149s 8ms/sample - loss: 0.1470
Epoch 4/5
19584/19584 [==============================] - 148s 8ms/sample - loss: 0.1320
Epoch 5/5
19584/19584 [==============================] - 147s 7ms/sample - loss: 0.1190
```

## 모델 구조
직전 모델과 동일

## 테스트 결과 및 총평
![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/82444b3a-29ed-49ba-9782-6412a34c52a6)

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/77b5d999-17af-47ba-86bc-a80efc25c36c)

직전 모델 (MSE Loss 가중치 2.5인 모델) 에 비해 잎이 녹색으로 표시되는 정도가 작으므로, **직전 모델보다 성능이 좋지 않다** 고 할 수 있다.

# Model 2 : skip connection 추가 적용 (2024.03.27 21시)
* 처음 **100장** 의 이미지만 이용하여 학습
* epoch **5회**
* MSE Loss 가중치 : **2.5**
* 일부 이미지에 대해 빨간색으로 칠해진 부분 근처에서 색상 그라데이션이 어색한 부분은 딥러닝 모델이 아닌 **test image 생성 알고리즘** 이슈이며 현재 해결한 상태임.

## 직전 모델 (Model 2) 대비 변동 사항
**2024.03.27 20시** (SiLU activation function 적용) 에서, Encoder 및 Decoder 로 입력되는 14 x 14 이미지의 정보 (총 8개) 를 flatten 해서 concatenated layer 로 직접 전달하는 **skip connection** 을 추가하여 적용

## 학습 로그
* 최종 Loss : **0.0821**

```
2024-03-27 21:22:03.239911: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
19584/19584 [==============================] - 154s 8ms/sample - loss: 0.3095
Epoch 2/5
19584/19584 [==============================] - 158s 8ms/sample - loss: 0.1171
Epoch 3/5
19584/19584 [==============================] - 155s 8ms/sample - loss: 0.1071
Epoch 4/5
19584/19584 [==============================] - 155s 8ms/sample - loss: 0.0938
Epoch 5/5
19584/19584 [==============================] - 153s 8ms/sample - loss: 0.0821
```

## 모델 구조
* Encoder 파라미터 개수 **601,776 개**
* Decoder 파라미터 개수 **602,082 개**
* 전체 모델 파라미터 개수 **1,203,858 개**

**Encoder 구조**

![encoder](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/d6ad4c3a-9b53-4baf-bb1e-2535b9bbb579)

**Decoder 구조**

![decoder](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/4e194426-d760-4dca-af0e-f360170c2398)

**VAE 기반 모델의 전체 구조**

![vae](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/ba965c7a-8af6-444e-a7c6-a6b8f80a073d)

## 테스트 결과

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/38c16b61-f88d-466d-838a-e262e80dd0e8)

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/58678383-c339-445f-90e6-a9c7c3505162)

# Model 1 : SiLU activation function 적용 (2024.03.27 20시)
* 처음 **100장** 의 이미지만 이용하여 학습
* epoch **5회**
* MSE Loss 가중치 : **4.0**

## 학습 로그
* 최종 Loss : **0.1634**

```
Train on 19584 samples
Epoch 1/5
2024-03-27 20:29:07.609220: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-03-27 20:29:07.834194: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
19584/19584 [==============================] - 131s 7ms/sample - loss: 0.5091
Epoch 2/5
19584/19584 [==============================] - 140s 7ms/sample - loss: 0.1812
Epoch 3/5
19584/19584 [==============================] - 150s 8ms/sample - loss: 0.1710
Epoch 4/5
19584/19584 [==============================] - 168s 9ms/sample - loss: 0.1664
Epoch 5/5
19584/19584 [==============================] - 145s 7ms/sample - loss: 0.1634
```
