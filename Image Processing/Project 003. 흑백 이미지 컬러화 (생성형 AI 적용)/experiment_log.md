# Model 6 : input positional info 추가 (2024.03.28 22시)
* 처음 **100장** 의 이미지만 이용하여 학습
* epoch **5회**
* MSE Loss 가중치 : **2.5**

## 직전 모델 (Model 5) 대비 변동 사항
* positional info를 기존 x좌표 상댓값 (0~1 범위), y좌표 상댓값 (0~1 범위) 외에 다음을 추가
  * 4개의 각 corner point와의 거리의 상댓값 (0~1 범위)
  * center point와의 거리의 상댓값 (0~1 범위)
* 모든 상댓값은 각 point에서의 해당 값을 모든 point에서의 해당 값들을 기준으로 min-max normalization 시킨 값임

## 학습 로그
* 최종 Loss : **0.0796** (Model 2 대비 **3.0%** 감소)

```
2024-03-28 22:05:47.489291: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
19584/19584 [==============================] - 125s 6ms/sample - loss: 0.3032
Epoch 2/5
19584/19584 [==============================] - 130s 7ms/sample - loss: 0.1140
Epoch 3/5
19584/19584 [==============================] - 124s 6ms/sample - loss: 0.1010
Epoch 4/5
19584/19584 [==============================] - 135s 7ms/sample - loss: 0.0896
Epoch 5/5
19584/19584 [==============================] - 136s 7ms/sample - loss: 0.0796
```

## 모델 구조
* Encoder 파라미터 개수 **729,712 개** (Model 4, 5 대비 0.04% 증가)
* Decoder 파라미터 개수 **730,178 개** (Model 4, 5 대비 0.07% 증가)
* 전체 모델 파라미터 개수 **1,459,890 개** (Model 4, 5 대비 0.05% 증가)

**Encoder 구조**

![encoder](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/90e56601-2444-4718-89f5-91665c1ba795)

**Decoder 구조**

![decoder](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/a4fc29e5-f2a8-4b5d-8485-55b6c3e941e8)

**전체 모델 구조**

![vae](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/b81bd569-9338-473c-8651-dd00df0f6ab1)

## 테스트 결과 및 총평

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/7eda8341-31fb-4416-a467-fb38dbbe31cd)

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/e17f8022-7455-4680-968a-cdbaa3c915c1)

전반적으로 Model 2 와 유사한 수준의 성능을 보인다.

# Model 5 : Model 4와 동일, 전체 이미지 및 epoch 16회 (2024.03.28 05시)
* **전체** 이미지를 대상으로 학습
* epoch **16회**
* MSE Loss 가중치 : **2.5**

## 직전 모델 (Model 4) 대비 변동 사항
없음

## 학습 로그
* 최종 Loss : **0.0536**

```
Epoch 1/16
2024-03-28 00:42:06.268189: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-03-28 00:42:06.492955: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
176192/176192 [==============================] - 1095s 6ms/sample - loss: 0.1339
Epoch 2/16
176192/176192 [==============================] - 1103s 6ms/sample - loss: 0.0885
Epoch 3/16
176192/176192 [==============================] - 1101s 6ms/sample - loss: 0.0798
Epoch 4/16
176192/176192 [==============================] - 1124s 6ms/sample - loss: 0.0744
Epoch 5/16
176192/176192 [==============================] - 1093s 6ms/sample - loss: 0.0704
Epoch 6/16
176192/176192 [==============================] - 1096s 6ms/sample - loss: 0.0663
Epoch 7/16
176192/176192 [==============================] - 1105s 6ms/sample - loss: 0.0639
Epoch 8/16
176192/176192 [==============================] - 1096s 6ms/sample - loss: 0.0623
Epoch 9/16
176192/176192 [==============================] - 1095s 6ms/sample - loss: 0.0602
Epoch 10/16
176192/176192 [==============================] - 1126s 6ms/sample - loss: 0.0628
Epoch 11/16
176192/176192 [==============================] - 1144s 6ms/sample - loss: 0.0580
Epoch 12/16
176192/176192 [==============================] - 1173s 7ms/sample - loss: 0.0566
Epoch 13/16
176192/176192 [==============================] - 1136s 6ms/sample - loss: 0.0560
Epoch 14/16
176192/176192 [==============================] - 1123s 6ms/sample - loss: 0.0551
Epoch 15/16
176192/176192 [==============================] - 1094s 6ms/sample - loss: 0.0545
Epoch 16/16
176192/176192 [==============================] - 1096s 6ms/sample - loss: 0.0536
```

## 테스트 결과 및 총평

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/f25a905a-59b0-46a9-bb65-16f49a350bd8)

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/82580c3d-1e32-457f-b6ed-88328b5b7388)

전체 이미지 사용 및 학습 epoch의 증가 등으로 **녹색 영역이 이전 모델들에 비해 늘어난 점** 은 긍정적으로 평가된다. 단, 빨간색으로 칠해진 부분 (장미) 이 직사각형 또는 정사각형에 가까운데, 해당 부분을 수정해야 한다.

# Model 4 : concatenate 직후 layer 뉴런 96 -> 128개 및 MSE Loss 가중치 2.0 적용 (2024.03.28 0시)
* 처음 **100장** 의 이미지만 이용하여 학습
* epoch **5회**
* MSE Loss 가중치 : **2.0**

## 직전 모델 (Model 3) 대비 변동 사항
* MSE Loss 가중치를 **4.0 -> 2.0** 으로 조정
* skip connection 직후에 이어지는 concatenate layer 의 직후의 layer 뉴런 개수를 **96개 -> 128개** 로 증가
  * 해당 레이어: ```{en,de}_lv{0,1,2,3}_5``` (총 8개)

## 학습 로그
* 최종 Loss : **0.0732**

```
2024-03-27 23:52:02.976041: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
19584/19584 [==============================] - 108s 6ms/sample - loss: 0.2842
Epoch 2/5
19584/19584 [==============================] - 114s 6ms/sample - loss: 0.0987
Epoch 3/5
19584/19584 [==============================] - 119s 6ms/sample - loss: 0.0921
Epoch 4/5
19584/19584 [==============================] - 121s 6ms/sample - loss: 0.0830
Epoch 5/5
19584/19584 [==============================] - 137s 7ms/sample - loss: 0.0732
```

## 모델 구조
* Encoder 파라미터 개수 **729,392 개** (Model 2, 3 대비 21.2% 증가)
* Decoder 파라미터 개수 **729,698 개** (Model 2, 3 대비 21.2% 증가)
* 전체 모델 파라미터 개수 **1,459,090 개** (Model 2, 3 대비 21.2% 증가)

**Encoder 구조**

![encoder](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/fbd441c1-4a09-4234-92bf-320380452a83)

**Decoder 구조**

![decoder](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/cd18bd13-ffb6-4001-b07c-d7b6dc982037)

**VAE 기반 모델의 전체 구조**

![vae](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/a42fc7f6-e28d-46ae-87c3-3560214503a8)

## 테스트 결과 및 총평

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/899338cb-cd3c-4511-a258-7e830a7b0d16)

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/db8934d3-48f5-4efb-8ac1-0924d35df372)

Model 2와 Model 3의 중간 정도의 성능을 보인다.

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

## 직전 모델 (Model 1) 대비 변동 사항
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
