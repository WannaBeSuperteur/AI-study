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