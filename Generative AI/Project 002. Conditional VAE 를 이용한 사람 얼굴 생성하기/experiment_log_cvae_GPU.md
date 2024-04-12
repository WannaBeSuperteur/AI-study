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