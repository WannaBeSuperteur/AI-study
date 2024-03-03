# Model 3 (2024.03.04 02:29)
## 이전 모델과의 차이점
* vs Model 2 : epochs 8 -> 40

## 실험 로그
* epochs : **40**
* 최종 train loss : **22.5267**

```
Epoch 1/40
2024-03-03 23:33:18.444728: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-03-03 23:33:18.515616: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
60000/60000 [==============================] - 254s 4ms/sample - loss: 33.6917
Epoch 2/40
60000/60000 [==============================] - 301s 5ms/sample - loss: 27.3895
Epoch 3/40
60000/60000 [==============================] - 338s 6ms/sample - loss: 26.0121
Epoch 4/40
60000/60000 [==============================] - 355s 6ms/sample - loss: 25.3619
Epoch 5/40
60000/60000 [==============================] - 355s 6ms/sample - loss: 24.9286
Epoch 6/40
60000/60000 [==============================] - 348s 6ms/sample - loss: 24.6144
Epoch 7/40
60000/60000 [==============================] - 354s 6ms/sample - loss: 24.4193
Epoch 8/40
60000/60000 [==============================] - 352s 6ms/sample - loss: 24.1963
Epoch 9/40
60000/60000 [==============================] - 349s 6ms/sample - loss: 24.0820
Epoch 10/40
60000/60000 [==============================] - 326s 5ms/sample - loss: 23.9116

...

Epoch 31/40
60000/60000 [==============================] - 239s 4ms/sample - loss: 22.7391
Epoch 32/40
60000/60000 [==============================] - 239s 4ms/sample - loss: 22.7207
Epoch 33/40
60000/60000 [==============================] - 240s 4ms/sample - loss: 22.6838
Epoch 34/40
60000/60000 [==============================] - 240s 4ms/sample - loss: 22.6657
Epoch 35/40
60000/60000 [==============================] - 240s 4ms/sample - loss: 22.6443
Epoch 36/40
60000/60000 [==============================] - 240s 4ms/sample - loss: 22.6055
Epoch 37/40
60000/60000 [==============================] - 240s 4ms/sample - loss: 22.5988
Epoch 38/40
60000/60000 [==============================] - 240s 4ms/sample - loss: 22.5587
Epoch 39/40
60000/60000 [==============================] - 240s 4ms/sample - loss: 22.5566
Epoch 40/40
60000/60000 [==============================] - 240s 4ms/sample - loss: 22.5267
```

## 실험 결과 및 총평
![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/4c81eac8-99c4-4e6a-830f-d099ac377a48)

* 이미지의 선명도에는 큰 차이가 없는 듯하다. (TODO : 회색 픽셀의 정도를 loss로 하는 아이디어?)
* 일부 이미지가 어색하다.

# Model 2 (2024.03.03 23:18)
## 이전 모델과의 차이점
* white 성분 비율에 따른 condition 값을 전체 데이터셋이 아닌, **각 숫자 class (0, 1, ..., 9) 별 평균 및 표준편차를 이용** 하는 것으로 변경

## 실험 로그
* epochs : **8**
* 최종 train loss : **24.1930**

```
Epoch 1/8
2024-03-03 22:45:54.238963: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-03-03 22:45:54.286821: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:354] MLIR V1 optimization pass is not enabled
60000/60000 [==============================] - 269s 4ms/sample - loss: 34.0153
Epoch 2/8
60000/60000 [==============================] - 239s 4ms/sample - loss: 27.4517
Epoch 3/8
60000/60000 [==============================] - 253s 4ms/sample - loss: 26.0133
Epoch 4/8
60000/60000 [==============================] - 241s 4ms/sample - loss: 25.3609
Epoch 5/8
60000/60000 [==============================] - 243s 4ms/sample - loss: 24.9078
Epoch 6/8
60000/60000 [==============================] - 240s 4ms/sample - loss: 24.6133
Epoch 7/8
60000/60000 [==============================] - 240s 4ms/sample - loss: 24.3711
Epoch 8/8
60000/60000 [==============================] - 240s 4ms/sample - loss: 24.1930
```

## 실험 결과 및 총평
![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/ede2fca0-29e0-4724-8572-8a5919090944)

* Model 1의 경우 숫자의 굵기를 설정하는 white ratio condition의 값이 동일함에도 '1'과 같은 특정 숫자가 다른 숫자보다 굵게 나오는 이슈가 있었는데, 해당 이슈가 해결되었다.
  * Model 1을 이용하여 해당 특정 숫자 이미지를 생성한 것이 다소 어색했는데, 이 점도 해결되었다.
* 숫자의 굵기를 설정하는 condition의 값은 잘 반영된 편이다.

# Model 1 (2024.03.03 22:00)
## 이전 모델과의 차이점
* 같은 latent vector를 통해 생성하는 숫자의 굵기를 일정하게 하기 위해, **숫자 이미지의 white 성분 비율** 을 C-VAE 모델의 추가 condition으로 지정
  * **white 성분 비율** : 이미지의 각 픽셀의 **white 성분** 값 (0~255) 을 255로 나눈 값 (검은색 0.0 ~ 회색 0.5 ~ 흰색 1.0) 의 평균
    * 해당 값이 1이면 완전한 흰색 이미지, 0이면 완전한 검은색 이미지이다.
    * MNIST 숫자 이미지의 경우, 실제로 확인하지는 않았지만 이 값이 평균 0.1 정도인 것으로 추측된다.

## 실험 로그
* epochs : **8**
* 최종 train loss : **24.85**

## 실험 결과 및 총평
![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/8ec916f7-29cb-45b7-9450-6ec44752c6ad)

* 위쪽 1-3, 4-6, 7-9, 10-12 번째 줄 각각이 같은 latent vector 값을 이용, 각 줄마다 굵기만 다름
* 숫자의 굵기를 설정하는 condition의 값이 실제로 잘 반영된 듯하다.
* 같은 latent vector를 사용하고 굵기 condition만 다르게 설정한 경우, 동일한 숫자들은 글자 스타일이 동일한 것을 확인할 수 있다.