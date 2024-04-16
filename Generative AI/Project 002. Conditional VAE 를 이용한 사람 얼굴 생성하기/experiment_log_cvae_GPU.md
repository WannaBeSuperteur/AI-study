## 실험 개요
* 실험 조건
  * 기본 설정
    * **all 16,864 images / 24 epochs**
    * **GPU에서 작동**
* 용어 설명
  * **HIDDEN_DIMS** : latent vector 차원 개수
  * **MSE Loss weight** : KL Loss의 weight에 대한 reconstruction loss (MSE Loss) 의 weight의 전체 loss에서의 비중
  * **info** : conditional VAE의 condition에 해당하는 값
    * **성별 정보** (male prob, female prob)
    * **hair color**
    * **mouth** (입을 벌린 정도)
    * **eyes** (눈을 뜬 정도)

## MODEL 19 (2024.04.16 22시, 23시)
* 직전 모델과의 차이점 : **모든 레이어의 activation function을 SiLU -> x * sigmoid(2x) 로 변경**
* HIDDEN_DIMS : **76**
* MSE loss weight : **200,000 (=200K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**

### MODEL 19 - 2차 (2024.04.16 23시)
* 최종 loss : **????.????** (1차 대비 **+/-??.????**)

```

```

### MODEL 19 - 1차 (2024.04.16 22시)
* 최종 loss : **2793.8148**

```
Epoch 1/24
2024-04-16 22:07:03.790457: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-16 22:07:04.630091: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 117s 7ms/sample - loss: 5659.1140 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 116s 7ms/sample - loss: 4118.5223 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3782.6401 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3617.1356 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3507.3642 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 118s 7ms/sample - loss: 3439.7082 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3369.1759 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3304.3703 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3254.5742 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3201.6890 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 116s 7ms/sample - loss: 3162.6207 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 116s 7ms/sample - loss: 3137.6773 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3094.0071 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3058.0974 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3031.6946 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3002.9558 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 116s 7ms/sample - loss: 2973.6291 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 2950.8904 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 116s 7ms/sample - loss: 2920.3257 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 116s 7ms/sample - loss: 2892.0231 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 2866.6380 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 116s 7ms/sample - loss: 2847.5805 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 2824.4828 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 2793.8148 - lr: 2.4108e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/e65721ad-f218-49ba-848a-ad90e30af0d4)

## MODEL 18 (2024.04.15 23시)
* 직전 모델과의 차이점 : **decoder의 마지막 (최종 이미지) 레이어의 activation function을 SiLU -> Sigmoid 로 변경**
* HIDDEN_DIMS : **76**
* MSE loss weight : **200,000 (=200K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**
* 최종 loss : **2930.6026**

```
Epoch 1/24
2024-04-15 23:22:02.942113: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-15 23:22:03.985999: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 116s 7ms/sample - loss: 5851.3410 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 114s 7ms/sample - loss: 4225.4568 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3898.3652 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3707.1438 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3610.1495 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3530.4510 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3443.7737 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3394.1050 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3341.2719 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3298.8508 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3266.6334 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3221.5336 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3194.8075 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3162.4549 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3139.8275 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3105.3737 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3081.0006 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3055.2277 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3031.9940 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 3014.1341 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 2990.3230 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 2971.2686 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 2950.3154 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 115s 7ms/sample - loss: 2930.6026 - lr: 2.4108e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/692837c0-01b8-4fc2-9bef-7a2bbc21c140)

## MODEL 17 (2024.04.15 0시)
* 직전 모델과의 차이점 : **decoder의 Add 레이어를 모두 Concatenate 레이어로 변경**
* HIDDEN_DIMS : **76**
* MSE loss weight : **200,000 (=200K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**
* 최종 loss : **3026.7852**
* 결론
  * MODEL 16에 비해, 학습 소요시간은 약간 증가한 반면 성능은 다소 떨어졌음.
  * 그러나, 학습 후반부 Loss 감소 속도가 MODEL 16 보다 약간 빠르므로, **롤백하지 않고 변경 사항 유지**

```
Epoch 1/24
2024-04-15 00:16:03.943640: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-15 00:16:04.812802: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 108s 6ms/sample - loss: 7774.0043 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 4831.7485 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 4209.3059 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 3979.2446 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 3828.8445 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3749.2516 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3673.4187 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 3593.2687 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3531.0243 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3487.3942 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3434.3376 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3390.6800 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3352.8429 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3309.3798 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 3275.5485 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 3231.9256 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 3199.9124 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 3174.4750 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 3141.1603 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 3117.9330 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3090.3221 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3072.4518 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3046.2044 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 3026.7852 - lr: 2.4108e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/0a61b83d-69cd-47ef-83c4-4c1f2c079c18)

## MODEL 16 (2024.04.14 21시)
* 직전 모델과의 차이점 : **Encoder와 Decoder의 dropout을 모두 0.45 -> 0.25 로 변경**
* HIDDEN_DIMS : **76**
* MSE loss weight : **200,000 (=200K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**
* 최종 loss : **2988.9638**

```
Epoch 1/24
2024-04-14 21:15:07.895327: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-14 21:15:08.749414: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 102s 6ms/sample - loss: 7915.9328 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 99s 6ms/sample - loss: 4985.0541 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 100s 6ms/sample - loss: 4274.8382 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3946.5499 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3779.0058 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 100s 6ms/sample - loss: 3670.6872 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3593.6391 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3507.3217 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3446.8128 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3382.7368 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3334.4691 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3290.1822 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 100s 6ms/sample - loss: 3257.7430 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3229.1561 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3183.9288 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 100s 6ms/sample - loss: 3165.8379 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 100s 6ms/sample - loss: 3133.8162 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3117.2851 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3092.2077 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 100s 6ms/sample - loss: 3062.3372 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 101s 6ms/sample - loss: 3051.4530 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 100s 6ms/sample - loss: 3035.8816 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 100s 6ms/sample - loss: 3011.6711 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 100s 6ms/sample - loss: 2988.9638 - lr: 2.4108e-04
```

**epoch에 따른 loss 그래프**

![cvae_train_result](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/c74e821a-7bb4-4616-be0e-19dc92f1007b)

**예시 생성 이미지**

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/41f5493e-eb0e-45cf-9a76-2aba0b2ece8c)

## MODEL 15 (2024.04.13 22시, 2024.04.14 20시)
* 직전 모델과의 차이점
  * MODEL 14에 적용한 다음 사항을 롤백
    * Encoder와 Decoder의 모든 레이어에 대해 2x2 Conv, 4x4 Conv를 각각 적용 후 add 하는 방법 사용
  * **MSE loss weight 을 50,000 (=50K) 에서 200,000 (=200K) 로 증가**
* HIDDEN_DIMS : **76**
* MSE loss weight : **200,000 (=200K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**

### MODEL 15 - 2차 (2024.04.14 20시)
* 최종 loss : **3028.6644** (1차 대비 **+6.5001**)

```
Epoch 1/24
2024-04-14 20:25:03.766942: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-14 20:25:04.633401: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 94s 6ms/sample - loss: 7822.3527 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 91s 5ms/sample - loss: 5043.3541 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 92s 5ms/sample - loss: 4411.3687 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 92s 5ms/sample - loss: 4068.0213 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 96s 6ms/sample - loss: 3877.4246 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3766.5778 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3660.5604 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3588.7366 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3518.6560 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3467.6519 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3407.0355 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3357.4831 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3321.0608 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3284.7523 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3246.4944 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3219.5606 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3195.0451 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3164.3190 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3140.7976 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3112.4717 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 3092.6237 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3069.0139 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3054.2028 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3028.6644 - lr: 2.4108e-04
```

**epoch에 따른 loss 그래프**

![cvae_train_result](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/27247f15-1cdc-4dad-bd7c-8dda74ce2f33)

**예시 생성 이미지**

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/e55b8a5c-95a8-4c53-88e1-c61ab014fb25)

### MODEL 15 - 1차 (2024.04.13 22시)
* 최종 loss : **3022.1643**

```
Epoch 1/24
2024-04-13 22:51:34.607773: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-13 22:51:35.535477: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 95s 6ms/sample - loss: 7655.5300 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 4911.2231 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 4320.9183 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 4016.2626 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3839.8141 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3711.2164 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3611.9575 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3543.3153 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3488.5959 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3425.6122 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3376.2684 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3329.2527 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3296.5806 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3264.1910 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3231.6491 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3199.4159 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3166.2485 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3149.7417 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3120.7769 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3095.5992 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3069.7995 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3053.0023 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3041.3783 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 3022.1643 - lr: 2.4108e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/12cd35c9-8cdf-4f8f-ab5b-094111b135da)

## MODEL 14 (2024.04.13 21시)
* 직전 모델과의 차이점
  * MODEL 13에 적용한 다음 사항을 롤백
    * Decoder의 각 레이어 (마지막 레이어를 제외한 3개) 마다 다음을 평균하는 레이어 추가
  * **Encoder와 Decoder의 모든 레이어에 대해 2x2 Conv, 4x4 Conv를 각각 적용 후 add** 하는 방법 사용 (모두 same padding)
* HIDDEN_DIMS : **76**
* MSE loss weight : **50000.0 (=50K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**
* 최종 loss : **774.8997**
* 실험 결과 및 결론
  * 2x2 Conv, 4x4 Conv add 적용에도 불구하고 MODEL 12, MODEL 13 과 오차 범위 내의 최종 loss를 보임
  * MODEL 13, MODEL 14 에서의 변경 사항을 모두 취소하고, **해당 변경 사항들에 대해서는 MODEL 12로 롤백 결정**

```
Epoch 1/24
2024-04-13 21:06:35.463404: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-13 21:06:36.338514: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 112s 7ms/sample - loss: 1866.8873 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 1214.6325 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 1080.1401 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 1008.2583 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 110s 7ms/sample - loss: 966.6472 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 936.5013 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 110s 7ms/sample - loss: 917.2839 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 110s 7ms/sample - loss: 896.7678 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 881.2158 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 110s 7ms/sample - loss: 869.6206 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 858.2306 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 847.8867 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 110s 7ms/sample - loss: 837.4093 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 830.7097 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 822.7494 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 817.6014 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 811.0067 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 805.7911 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 797.7601 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 793.3352 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 788.5164 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 782.1806 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 777.5712 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 774.8997 - lr: 2.4108e-04
```

**Encoder 모델 구조**

![encoder](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/bd6e4063-1282-4f28-925d-0fd5ac1d0260)

**Decoder 모델 구조**

![decoder](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/fa8c4e73-b671-458d-aa0d-4430a5d6ee75)

**예시 생성 이미지**

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/22e14690-1c79-4d43-b15b-91a7f476fd30)

## MODEL 13 (2024.04.13 17시)
* 직전 모델과의 차이점
  * Decoder의 각 레이어 (마지막 레이어를 제외한 3개) 마다 다음을 평균하는 레이어 추가
    * identity
    * 2x2 Kernel 의 2D Convolution 레이어
    * 3x3 Kernel 의 2D Convolution 레이어
    * 4x4 Kernel 의 2D Convolution 레이어
  * 상기한 2x2, 3x3, 4x4의 2D convolution layer 는 모두 same padding 적용  
* HIDDEN_DIMS : **76**
* MSE loss weight : **50000.0 (=50K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**
* 최종 loss : **771.1474**
* 특징
  * 모델 구조가 MODEL 12 보다 복잡해졌기 때문에 학습 시간 증가 (전체 24 epochs 기준 37분 -> 1시간 3분)
  * 모델 구조가 복잡해졌음에도 **최종 loss는 오차 범위를 고려할 때 MODEL 12와 사실상 차이가 없으므로, 본 변경은 적용하지 않음**

```
Epoch 1/24
2024-04-13 17:19:57.078688: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-13 17:19:57.981503: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 157s 9ms/sample - loss: 1969.1082 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 155s 9ms/sample - loss: 1239.9702 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 157s 9ms/sample - loss: 1081.2427 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 157s 9ms/sample - loss: 1002.9044 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 961.4564 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 929.7188 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 159s 9ms/sample - loss: 908.4392 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 159s 9ms/sample - loss: 893.0965 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 877.4231 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 159s 9ms/sample - loss: 864.2533 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 852.4276 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 844.9243 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 839.3536 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 830.7213 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 823.5034 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 815.9373 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 809.0727 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 801.9664 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 795.9257 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 791.3182 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 787.1661 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 782.1028 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 777.0560 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 158s 9ms/sample - loss: 771.1474 - lr: 2.4108e-04
```

**Decoder 모델 구조**

![decoder](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/5aeb7847-e28b-48be-89e1-282388a3069f)

**예시 생성 이미지**

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/206da9f2-e604-4024-b93a-3ffbc90542ce)

## MODEL 12 (2024.04.13 15시)
* 직전 모델과의 차이점
  * Encoder와 Decoder의 각 layer 의 필터 개수를 MODEL 9 와 동일한 값으로 롤백
  * Decoder의 마지막 레이어 ```Conv2DTranspose``` (2D Transpose Convolution) 를 ```Conv2D``` (2D Convolution Layer) 레이어로 변경
* HIDDEN_DIMS : **76**
* MSE loss weight : **50000.0 (=50K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**
* 최종 loss : **774.4656**

```
Epoch 1/24
2024-04-13 15:41:32.792294: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-13 15:41:33.719264: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 95s 6ms/sample - loss: 1905.8512 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 93s 5ms/sample - loss: 1238.7471 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 93s 5ms/sample - loss: 1086.4069 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 999.7828 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 957.5870 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 931.1225 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 904.0660 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 889.0775 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 874.9561 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 863.6313 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 854.3229 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 845.4591 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 835.0798 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 831.4741 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 821.5918 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 816.5080 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 811.6130 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 802.8952 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 796.9773 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 794.4094 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 787.4577 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 782.8555 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 93s 6ms/sample - loss: 779.5396 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 94s 6ms/sample - loss: 774.4656 - lr: 2.4108e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/2bf93d96-ebc0-425c-b8a0-92ea2734cd83)

## MODEL 11 (2024.04.13 14시)
* 직전 모델과의 차이점 : Encoder 및 Decoder의 각 layer의 필터 개수 증가
* HIDDEN_DIMS : **76**
* MSE loss weight : **50000.0 (=50K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**
* 최종 loss : **800.0506**
* 특징 :
  * 각 layer의 filter 개수를 늘린 결과, MODEL 10에 비해 학습 시간 약간 증가 (전체 24 epochs에 대한 학습 시간 약 36분 -> 43분 예상)
  * **Filter의 개수를 늘렸음에도 불구하고 최종 loss가 MODEL 9, MODEL 10에 비해 모두 증가하여 성능이 떨어짐 -> Filter 개수는 MODEL 12 부터는 MODEL 9와 동일한 값으로 롤백 결정**

```
Epoch 1/24
2024-04-13 14:11:21.103318: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-13 14:11:22.125579: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 111s 7ms/sample - loss: 2050.9387 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 1282.4948 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 1135.4803 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 1072.9134 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 1031.7102 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 1002.7163 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 970.3709 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 948.0155 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 926.7214 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 909.3596 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 898.4523 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 887.5808 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 878.4997 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 866.3294 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 858.5767 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 850.7181 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 843.0845 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 835.4311 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 829.2441 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 820.9955 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 816.0943 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 810.3657 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 802.9045 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 800.0506 - lr: 2.4108e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/9e3f31a2-e8b9-4ac4-ae45-09d3afa24705)

## MODEL 10 (2024.04.13 13시)
* 직전 모델과의 차이점 :
  * Encoder의 각 layer의 필터 개수 증가
  * Decoder의 각 layer의 필터 개수를 Encoder와 일치
* HIDDEN_DIMS : **76**
* MSE loss weight : **50000.0 (=50K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**
* 최종 loss : **788.8723**
* 특징 :
  * 각 layer의 filter 개수를 조정한 결과, MODEL 9에 비해 학습 시간 약간 감소 (전체 24 epochs에 대한 학습 시간 약 39분 -> 36분 예상)
  * MODEL 9에 비해, **Filter 개수가 전반적으로 늘어난 편인데 최종 loss는 오히려 소폭 (5.0 정도) 증가** -> 성능이 덜 좋음
    * 일부 레이어의 필터 개수가 MODEL 9에 비해 감소했으므로, Filter 개수를 Encoder, Decoder 모두 전체적으로 더 늘려서 시도. 그래도 성능이 좋아지지 않으면 Filter 개수는 MODEL 9와 같아지도록 롤백 예정

```
Epoch 1/24
2024-04-13 13:28:36.656520: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-13 13:28:37.510010: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 92s 5ms/sample - loss: 1954.1245 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 88s 5ms/sample - loss: 1210.9713 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 1080.4601 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 1031.9258 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 999.4374 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 91s 5ms/sample - loss: 971.7965 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 91s 5ms/sample - loss: 946.1510 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 926.8413 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 909.6772 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 893.9399 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 877.0321 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 868.0843 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 91s 5ms/sample - loss: 859.0896 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 850.1559 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 844.3859 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 835.7522 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 829.7302 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 822.6768 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 818.1995 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 812.2342 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 805.0776 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 799.3312 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 794.0649 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 90s 5ms/sample - loss: 788.8723 - lr: 2.4108e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/c6d761e3-81d5-4992-a8b0-da258002a099)

## MODEL 9 (2024.04.13 12시)
* 직전 모델과의 차이점 : **Learning rate 초기값 0.0002 -> 0.0004, 감소율 0.99 -> 0.975 로 변경**
* HIDDEN_DIMS : **76**
* MSE loss weight : **50000.0 (=50K)**
* learning rate 초기값 : **0.0004**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.975**
* 최종 loss : **783.4584**
* 특징 : Learning rate 설정을 변경하면서 MODEL 8에 비해 학습 시간 약 10% 증가 (전체 24 epochs에 대한 학습 시간 약 36분 -> 39분 예상)

```
Epoch 1/24
2024-04-13 12:37:20.623549: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-13 12:37:21.511882: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 99s 6ms/sample - loss: 1977.2901 - lr: 4.0000e-04
Epoch 2/24
16864/16864 [==============================] - 96s 6ms/sample - loss: 1285.7301 - lr: 4.0000e-04
Epoch 3/24
16864/16864 [==============================] - 96s 6ms/sample - loss: 1114.2525 - lr: 4.0000e-04
Epoch 4/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 1030.5190 - lr: 4.0000e-04
Epoch 5/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 983.8497 - lr: 3.9000e-04
Epoch 6/24
16864/16864 [==============================] - 99s 6ms/sample - loss: 955.3330 - lr: 3.8025e-04
Epoch 7/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 932.2115 - lr: 3.7074e-04
Epoch 8/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 912.2270 - lr: 3.6148e-04
Epoch 9/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 897.0994 - lr: 3.5244e-04
Epoch 10/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 883.9896 - lr: 3.4363e-04
Epoch 11/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 869.7460 - lr: 3.3504e-04
Epoch 12/24
16864/16864 [==============================] - 98s 6ms/sample - loss: 860.2677 - lr: 3.2666e-04
Epoch 13/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 851.4844 - lr: 3.1849e-04
Epoch 14/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 842.2469 - lr: 3.1053e-04
Epoch 15/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 836.4577 - lr: 3.0277e-04
Epoch 16/24
16864/16864 [==============================] - 98s 6ms/sample - loss: 827.4164 - lr: 2.9520e-04
Epoch 17/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 818.6436 - lr: 2.8782e-04
Epoch 18/24
16864/16864 [==============================] - 98s 6ms/sample - loss: 813.1003 - lr: 2.8062e-04
Epoch 19/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 807.1768 - lr: 2.7361e-04
Epoch 20/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 802.6254 - lr: 2.6677e-04
Epoch 21/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 796.2873 - lr: 2.6010e-04
Epoch 22/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 791.5585 - lr: 2.5360e-04
Epoch 23/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 786.2370 - lr: 2.4726e-04
Epoch 24/24
16864/16864 [==============================] - 97s 6ms/sample - loss: 783.4584 - lr: 2.4108e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/88336b16-4ef5-40d4-9f70-36b28f508c70)

## MODEL 8 (2024.04.13 11시)
* 직전 모델과의 차이점 : **Encoder와 Decoder의 모든 Convolutional Layer에 Batch Normalization 추가**
* HIDDEN_DIMS : **76**
* MSE loss weight : **50000.0 (=50K)**
* learning rate 초기값 : **0.0002**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.99**
* 최종 loss : **821.7564**
* 특징 : Batch Normalization을 적용하면서 MODEL 7 이전의 모델에 비해 학습 시간 약 20% 감소

```
Epoch 1/24
2024-04-13 11:43:04.227753: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-13 11:43:05.156341: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 90s 5ms/sample - loss: 2106.4139 - lr: 2.0000e-04
Epoch 2/24
16864/16864 [==============================] - 88s 5ms/sample - loss: 1260.2358 - lr: 2.0000e-04
Epoch 3/24
16864/16864 [==============================] - 88s 5ms/sample - loss: 1092.6693 - lr: 2.0000e-04
Epoch 4/24
16864/16864 [==============================] - 88s 5ms/sample - loss: 1027.5900 - lr: 2.0000e-04
Epoch 5/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 986.6379 - lr: 1.9800e-04
Epoch 6/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 962.1125 - lr: 1.9602e-04
Epoch 7/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 941.9920 - lr: 1.9406e-04
Epoch 8/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 927.2079 - lr: 1.9212e-04
Epoch 9/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 916.2241 - lr: 1.9020e-04
Epoch 10/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 905.7427 - lr: 1.8830e-04
Epoch 11/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 896.1881 - lr: 1.8641e-04
Epoch 12/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 885.8126 - lr: 1.8455e-04
Epoch 13/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 879.0626 - lr: 1.8270e-04
Epoch 14/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 871.8953 - lr: 1.8088e-04
Epoch 15/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 864.5778 - lr: 1.7907e-04
Epoch 16/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 857.7832 - lr: 1.7728e-04
Epoch 17/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 853.2042 - lr: 1.7550e-04
Epoch 18/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 849.7991 - lr: 1.7375e-04
Epoch 19/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 844.8091 - lr: 1.7201e-04
Epoch 20/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 837.7922 - lr: 1.7029e-04
Epoch 21/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 831.1557 - lr: 1.6859e-04
Epoch 22/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 828.8731 - lr: 1.6690e-04
Epoch 23/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 822.5771 - lr: 1.6523e-04
Epoch 24/24
16864/16864 [==============================] - 89s 5ms/sample - loss: 821.7564 - lr: 1.6358e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/551f48f7-628b-4bf3-afcf-991f169eb5a3)

## MODEL 7 (2024.04.13 9시)
* 직전 모델과의 차이점 : **MSE Loss weight을 8000.0 (=8K) 에서 50000.0 (=50K) 으로, HIDDEN_DIMS 를 40 -> 76 으로 변경**
* HIDDEN_DIMS : **76**
* MSE loss weight : **50000.0 (=50K)**
* learning rate 초기값 : **0.0002**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.99**
* 최종 loss : **855.9115**

```
Epoch 1/24
2024-04-13 09:33:33.749366: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-13 09:33:36.187204: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 117s 7ms/sample - loss: 2032.8014 - lr: 2.0000e-04
Epoch 2/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 1301.5614 - lr: 2.0000e-04
Epoch 3/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 1151.0361 - lr: 2.0000e-04
Epoch 4/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 1076.8823 - lr: 2.0000e-04
Epoch 5/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 1037.8377 - lr: 1.9800e-04
Epoch 6/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 1013.1902 - lr: 1.9602e-04
Epoch 7/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 995.7076 - lr: 1.9406e-04
Epoch 8/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 981.0785 - lr: 1.9212e-04
Epoch 9/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 968.7423 - lr: 1.9020e-04
Epoch 10/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 955.1183 - lr: 1.8830e-04
Epoch 11/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 945.8988 - lr: 1.8641e-04
Epoch 12/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 937.2372 - lr: 1.8455e-04
Epoch 13/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 926.8399 - lr: 1.8270e-04
Epoch 14/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 918.5990 - lr: 1.8088e-04
Epoch 15/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 910.6222 - lr: 1.7907e-04
Epoch 16/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 900.2229 - lr: 1.7728e-04
Epoch 17/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 895.7114 - lr: 1.7550e-04
Epoch 18/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 887.2233 - lr: 1.7375e-04
Epoch 19/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 879.9285 - lr: 1.7201e-04
Epoch 20/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 876.3959 - lr: 1.7029e-04
Epoch 21/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 869.2951 - lr: 1.6859e-04
Epoch 22/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 864.0885 - lr: 1.6690e-04
Epoch 23/24
16864/16864 [==============================] - 111s 7ms/sample - loss: 860.6324 - lr: 1.6523e-04
Epoch 24/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 855.9115 - lr: 1.6358e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/1cbc31c8-d6bc-4176-a661-3b4c1fbce20b)

## MODEL 6 (2024.04.13 0시)
* 직전 모델과의 차이점 : **MSE Loss weight을 2000.0 에서 8000.0 으로, HIDDEN_DIMS 를 20 -> 40 으로 변경**
* HIDDEN_DIMS : **40**
* MSE loss weight : **8000.0**
* learning rate 초기값 : **0.0002**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.99**
* 최종 loss : **175.9354**

```
Epoch 1/24
2024-04-13 00:29:36.070503: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-13 00:29:36.965401: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 112s 7ms/sample - loss: 360.4038 - lr: 2.0000e-04
Epoch 2/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 228.8320 - lr: 2.0000e-04
Epoch 3/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 208.9753 - lr: 2.0000e-04
Epoch 4/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 201.8692 - lr: 2.0000e-04
Epoch 5/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 197.5608 - lr: 1.9800e-04
Epoch 6/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 195.3923 - lr: 1.9602e-04
Epoch 7/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 192.8856 - lr: 1.9406e-04
Epoch 8/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 191.1724 - lr: 1.9212e-04
Epoch 9/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 189.6849 - lr: 1.9020e-04
Epoch 10/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 188.5741 - lr: 1.8830e-04
Epoch 11/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 187.3164 - lr: 1.8641e-04
Epoch 12/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 186.0530 - lr: 1.8455e-04
Epoch 13/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 185.4295 - lr: 1.8270e-04
Epoch 14/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 184.7548 - lr: 1.8088e-04
Epoch 15/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 183.2697 - lr: 1.7907e-04
Epoch 16/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 182.2947 - lr: 1.7728e-04
Epoch 17/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 181.4829 - lr: 1.7550e-04
Epoch 18/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 180.7224 - lr: 1.7375e-04
Epoch 19/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 180.0249 - lr: 1.7201e-04
Epoch 20/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 179.0817 - lr: 1.7029e-04
Epoch 21/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 178.0633 - lr: 1.6859e-04
Epoch 22/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 177.4755 - lr: 1.6690e-04
Epoch 23/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 176.7015 - lr: 1.6523e-04
Epoch 24/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 175.9354 - lr: 1.6358e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/8bd72718-ecc8-4f83-af11-ae2c52da02be)

## MODEL 5 (2024.04.12 23시)
* 직전 모델과의 차이점 : **MSE Loss weight을 500.0 에서 2000.0 으로 변경**
* HIDDEN_DIMS : **20**
* MSE loss weight : **2000.0**
* learning rate 초기값 : **0.0002**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.99**
* 최종 loss : **56.8773**

```
Epoch 1/24
2024-04-12 23:39:09.083580: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-12 23:39:09.944213: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 109s 6ms/sample - loss: 100.0565 - lr: 2.0000e-04
Epoch 2/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 64.8783 - lr: 2.0000e-04
Epoch 3/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 62.5575 - lr: 2.0000e-04
Epoch 4/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 61.4097 - lr: 2.0000e-04
Epoch 5/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 60.6282 - lr: 1.9800e-04
Epoch 6/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 60.1959 - lr: 1.9602e-04
Epoch 7/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 59.7177 - lr: 1.9406e-04
Epoch 8/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 59.5644 - lr: 1.9212e-04
Epoch 9/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 59.1830 - lr: 1.9020e-04
Epoch 10/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 58.8539 - lr: 1.8830e-04
Epoch 11/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 58.6779 - lr: 1.8641e-04
Epoch 12/24
16864/16864 [==============================] - 105s 6ms/sample - loss: 58.3101 - lr: 1.8455e-04
Epoch 13/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 58.3450 - lr: 1.8270e-04
Epoch 14/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 58.0988 - lr: 1.8088e-04
Epoch 15/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 57.9874 - lr: 1.7907e-04
Epoch 16/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 57.8993 - lr: 1.7728e-04
Epoch 17/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 57.6582 - lr: 1.7550e-04
Epoch 18/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 57.5783 - lr: 1.7375e-04
Epoch 19/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 57.4056 - lr: 1.7201e-04
Epoch 20/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 57.2363 - lr: 1.7029e-04
Epoch 21/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 57.1689 - lr: 1.6859e-04
Epoch 22/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 57.0561 - lr: 1.6690e-04
Epoch 23/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 56.9008 - lr: 1.6523e-04
Epoch 24/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 56.8773 - lr: 1.6358e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/6e25714a-6ccb-4fd4-a96d-417d41c729cb)

## MODEL 4 (2024.04.12 22시)
* 직전 모델과의 차이점 : **HIDDEN_DIMS 를 48 -> 20 으로 변경**
* HIDDEN_DIMS : **20**
* MSE loss weight : **500.0**
* learning rate 초기값 : **0.0002**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.99**
* 최종 loss : **18.2257**

```
Epoch 1/24
2024-04-12 22:49:31.569858: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-12 22:49:34.364734: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 117s 7ms/sample - loss: 28.9000 - lr: 2.0000e-04
Epoch 2/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 20.1185 - lr: 2.0000e-04
Epoch 3/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 19.5872 - lr: 2.0000e-04
Epoch 4/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 19.3550 - lr: 2.0000e-04
Epoch 5/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 19.1381 - lr: 1.9800e-04
Epoch 6/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 19.0479 - lr: 1.9602e-04
Epoch 7/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.9686 - lr: 1.9406e-04
Epoch 8/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.9217 - lr: 1.9212e-04
Epoch 9/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.8228 - lr: 1.9020e-04
Epoch 10/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.7811 - lr: 1.8830e-04
Epoch 11/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.6944 - lr: 1.8641e-04
Epoch 12/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.6768 - lr: 1.8455e-04
Epoch 13/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.5575 - lr: 1.8270e-04
Epoch 14/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.5139 - lr: 1.8088e-04
Epoch 15/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.4767 - lr: 1.7907e-04
Epoch 16/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.4315 - lr: 1.7728e-04
Epoch 17/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.3994 - lr: 1.7550e-04
Epoch 18/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.3713 - lr: 1.7375e-04
Epoch 19/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.3061 - lr: 1.7201e-04
Epoch 20/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.3408 - lr: 1.7029e-04
Epoch 21/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.2659 - lr: 1.6859e-04
Epoch 22/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.2547 - lr: 1.6690e-04
Epoch 23/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.2256 - lr: 1.6523e-04
Epoch 24/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.2257 - lr: 1.6358e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/63501838-f54b-498c-8ebd-73b1981f80de)

## MODEL 3 (2024.04.12 0시)
* 직전 모델과의 차이점 : **HIDDEN_DIMS 를 48 -> 110 으로 변경**
* HIDDEN_DIMS : **110**
* MSE loss weight : **500.0**
* learning rate 초기값 : **0.0002**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.99**
* 최종 loss : **18.7374**

```
Epoch 1/24
2024-04-12 00:08:36.019308: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-12 00:08:36.969610: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 116s 7ms/sample - loss: 28.2506 - lr: 2.0000e-04
Epoch 2/24
16864/16864 [==============================] - 113s 7ms/sample - loss: 21.0581 - lr: 2.0000e-04
Epoch 3/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 20.5806 - lr: 2.0000e-04
Epoch 4/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 20.3224 - lr: 2.0000e-04
Epoch 5/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 20.0537 - lr: 1.9800e-04
Epoch 6/24
16864/16864 [==============================] - 113s 7ms/sample - loss: 19.9224 - lr: 1.9602e-04
Epoch 7/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 19.7479 - lr: 1.9406e-04
Epoch 8/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 19.5718 - lr: 1.9212e-04
Epoch 9/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 19.4938 - lr: 1.9020e-04
Epoch 10/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 19.3433 - lr: 1.8830e-04
Epoch 11/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 19.2425 - lr: 1.8641e-04
Epoch 12/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 19.1931 - lr: 1.8455e-04
Epoch 13/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 19.1291 - lr: 1.8270e-04
Epoch 14/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 19.0327 - lr: 1.8088e-04
Epoch 15/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 18.9714 - lr: 1.7907e-04
Epoch 16/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 18.9318 - lr: 1.7728e-04
Epoch 17/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 18.9326 - lr: 1.7550e-04
Epoch 18/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 18.8881 - lr: 1.7375e-04
Epoch 19/24
16864/16864 [==============================] - 113s 7ms/sample - loss: 18.8077 - lr: 1.7201e-04
Epoch 20/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 18.8233 - lr: 1.7029e-04
Epoch 21/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 18.8247 - lr: 1.6859e-04
Epoch 22/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 18.8188 - lr: 1.6690e-04
Epoch 23/24
16864/16864 [==============================] - 113s 7ms/sample - loss: 18.7070 - lr: 1.6523e-04
Epoch 24/24
16864/16864 [==============================] - 112s 7ms/sample - loss: 18.7374 - lr: 1.6358e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/7650ed5d-de96-409d-9c9e-d099946484f8)

## MODEL 2 (2024.04.11 23시)
* 직전 모델과의 차이점 : **MSE loss weight을 1.0 -> 500.0 으로 변경**
* HIDDEN_DIMS : **48**
* MSE loss weight : **500.0**
* learning rate 초기값 : **0.0002**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.99**
* 최종 loss : **18.4235**

```
Epoch 1/24
2024-04-11 23:08:11.875221: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-11 23:08:12.814828: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 111s 7ms/sample - loss: 27.3757 - lr: 2.0000e-04
Epoch 2/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 20.5762 - lr: 2.0000e-04
Epoch 3/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 20.0954 - lr: 2.0000e-04
Epoch 4/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 19.8132 - lr: 2.0000e-04
Epoch 5/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 19.5952 - lr: 1.9800e-04
Epoch 6/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 19.4082 - lr: 1.9602e-04
Epoch 7/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 19.3063 - lr: 1.9406e-04
Epoch 8/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 19.2463 - lr: 1.9212e-04
Epoch 9/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 19.1464 - lr: 1.9020e-04
Epoch 10/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 19.0746 - lr: 1.8830e-04
Epoch 11/24
16864/16864 [==============================] - 109s 6ms/sample - loss: 18.9988 - lr: 1.8641e-04
Epoch 12/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.9168 - lr: 1.8455e-04
Epoch 13/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.8918 - lr: 1.8270e-04
Epoch 14/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.7896 - lr: 1.8088e-04
Epoch 15/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.7955 - lr: 1.7907e-04
Epoch 16/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.7680 - lr: 1.7728e-04
Epoch 17/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.7418 - lr: 1.7550e-04
Epoch 18/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.7346 - lr: 1.7375e-04
Epoch 19/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.6913 - lr: 1.7201e-04
Epoch 20/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.6395 - lr: 1.7029e-04
Epoch 21/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.6016 - lr: 1.6859e-04
Epoch 22/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.5383 - lr: 1.6690e-04
Epoch 23/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.5386 - lr: 1.6523e-04
Epoch 24/24
16864/16864 [==============================] - 108s 6ms/sample - loss: 18.4235 - lr: 1.6358e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/51fbd36d-9b88-4af7-adbb-109376b3b640)

## baseline model (2024.04.11 22시)
* HIDDEN_DIMS : **48**
* MSE loss weight : **1.0**
* learning rate 초기값 : **0.0002**
* epoch 4 이후 learning rate 감소율 (직전 epoch 대비 현재 epoch의 learning rate의 비율) : **0.99**
* 최종 loss : **0.0449**

```
Epoch 1/24
2024-04-11 22:19:53.274499: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-04-11 22:19:54.186695: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
16864/16864 [==============================] - 108s 6ms/sample - loss: 0.5556 - lr: 2.0000e-04
Epoch 2/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 0.1183 - lr: 2.0000e-04
Epoch 3/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0837 - lr: 2.0000e-04
Epoch 4/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0701 - lr: 2.0000e-04
Epoch 5/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0630 - lr: 1.9800e-04
Epoch 6/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0586 - lr: 1.9602e-04
Epoch 7/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 0.0555 - lr: 1.9406e-04
Epoch 8/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 0.0533 - lr: 1.9212e-04
Epoch 9/24
16864/16864 [==============================] - 117s 7ms/sample - loss: 0.0517 - lr: 1.9020e-04
Epoch 10/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0504 - lr: 1.8830e-04
Epoch 11/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0495 - lr: 1.8641e-04
Epoch 12/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 0.0487 - lr: 1.8455e-04
Epoch 13/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0480 - lr: 1.8270e-04
Epoch 14/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0474 - lr: 1.8088e-04
Epoch 15/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 0.0468 - lr: 1.7907e-04
Epoch 16/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0463 - lr: 1.7728e-04
Epoch 17/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 0.0458 - lr: 1.7550e-04
Epoch 18/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0455 - lr: 1.7375e-04
Epoch 19/24
16864/16864 [==============================] - 106s 6ms/sample - loss: 0.0453 - lr: 1.7201e-04
Epoch 20/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0451 - lr: 1.7029e-04
Epoch 21/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0450 - lr: 1.6859e-04
Epoch 22/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0450 - lr: 1.6690e-04
Epoch 23/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0449 - lr: 1.6523e-04
Epoch 24/24
16864/16864 [==============================] - 107s 6ms/sample - loss: 0.0449 - lr: 1.6358e-04
```

![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/0b8a2938-8d32-467c-8012-cf81e37f059a)