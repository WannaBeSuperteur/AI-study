## 모델
```regression_background_mean```

## 모델의 목적
이미지에 있는 인물 사진의 배경이 밝은 정도를 예측한다.
* 0부터 1까지의 값
* ```1.0``` : 배경이 밝음, ```0.5``` : 배경이 중간 밝기의 색임, ```0.0``` : 배경이 머리카락 색 정도 또는 그보다 더 어두움

## 모델 학습 로그
* 학습 설정
  * max epochs : **40 epochs**
  * 학습 데이터 개수 : 남성 1,000 장, 여성 1,000 장의 **총 2,000 장 이미지**
* 학습 결과 : 각 성별 평균값으로 예측했을 때에 비해 MSE 기준으로 오차가 **1.87배** 작음
  * validation loss (MSE) : **0.0490**
  * 각 성별에 대해 평균값 (male: 0.5045, female: 0.5025) 으로 예측했을 때의 MSE (male: 0.0902, female: 0.0932) 의 평균값 : **0.0917** 

```
train input (gender prob) : (2000, 2)
[[9.9998210e-01 1.7932965e-05]
 [9.1045230e-01 8.9547730e-02]
 [9.9990190e-01 9.8053760e-05]
 ...
 [1.3854587e-04 9.9986150e-01]
 [3.9756913e-09 1.0000000e+00]
 [4.5712350e-05 9.9995434e-01]]

train output              : (2000, 1)
[[1. ]
 [0.5]
 [1. ]
 ...
 [0. ]
 [0. ]
 [0.5]]

Epoch 1/40
57/57 [==============================] - 37s 630ms/step - loss: 0.2424 - val_loss: 0.1159 - lr: 0.0010
Epoch 2/40
57/57 [==============================] - 43s 761ms/step - loss: 0.0846 - val_loss: 0.0840 - lr: 0.0010
Epoch 3/40
57/57 [==============================] - 46s 799ms/step - loss: 0.0624 - val_loss: 0.0662 - lr: 0.0010
Epoch 4/40
57/57 [==============================] - 48s 836ms/step - loss: 0.0555 - val_loss: 0.0637 - lr: 0.0010
Epoch 5/40
57/57 [==============================] - 47s 821ms/step - loss: 0.0493 - val_loss: 0.0636 - lr: 0.0010
Epoch 6/40
57/57 [==============================] - 47s 832ms/step - loss: 0.0487 - val_loss: 0.0631 - lr: 0.0010
Epoch 7/40
57/57 [==============================] - 47s 826ms/step - loss: 0.0424 - val_loss: 0.0718 - lr: 0.0010
Epoch 8/40
57/57 [==============================] - 46s 814ms/step - loss: 0.0410 - val_loss: 0.0546 - lr: 0.0010
Epoch 9/40
57/57 [==============================] - 47s 820ms/step - loss: 0.0414 - val_loss: 0.0659 - lr: 0.0010
Epoch 10/40
57/57 [==============================] - 46s 814ms/step - loss: 0.0430 - val_loss: 0.0600 - lr: 0.0010
Epoch 11/40
57/57 [==============================] - 48s 833ms/step - loss: 0.0352 - val_loss: 0.0511 - lr: 1.0000e-04
Epoch 12/40
57/57 [==============================] - 46s 805ms/step - loss: 0.0323 - val_loss: 0.0497 - lr: 1.0000e-04
Epoch 13/40
57/57 [==============================] - 44s 779ms/step - loss: 0.0310 - val_loss: 0.0491 - lr: 1.0000e-04
Epoch 14/40
57/57 [==============================] - 48s 836ms/step - loss: 0.0292 - val_loss: 0.0485 - lr: 1.0000e-04
Epoch 15/40
57/57 [==============================] - 46s 814ms/step - loss: 0.0289 - val_loss: 0.0482 - lr: 1.0000e-04
Epoch 16/40
57/57 [==============================] - 45s 795ms/step - loss: 0.0274 - val_loss: 0.0480 - lr: 1.0000e-04
Epoch 17/40
57/57 [==============================] - 47s 826ms/step - loss: 0.0267 - val_loss: 0.0494 - lr: 1.0000e-04
Epoch 18/40
57/57 [==============================] - 46s 810ms/step - loss: 0.0263 - val_loss: 0.0490 - lr: 1.0000e-04
Epoch 19/40
57/57 [==============================] - 46s 814ms/step - loss: 0.0248 - val_loss: 0.0491 - lr: 1.0000e-05
Epoch 20/40
57/57 [==============================] - 47s 817ms/step - loss: 0.0251 - val_loss: 0.0490 - lr: 1.0000e-05
Epoch 21/40
57/57 [==============================] - 46s 805ms/step - loss: 0.0246 - val_loss: 0.0490 - lr: 1.0000e-06
Model: "regression__background__mean__model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           multiple                  0

 max_pooling2d (MaxPooling2D  multiple                 0
 )

 dropout (Dropout)           multiple                  0

 conv2d (Conv2D)             multiple                  896

 conv2d_1 (Conv2D)           multiple                  9248

 conv2d_2 (Conv2D)           multiple                  18496

 conv2d_3 (Conv2D)           multiple                  36928

 conv2d_4 (Conv2D)           multiple                  36928

 dense_0 (Dense)             multiple                  1982720

 dense_1 (Dense)             multiple                  16576

 dense_final (Dense)         multiple                  65

=================================================================
Total params: 2,101,857
Trainable params: 2,101,857
Non-trainable params: 0
_________________________________________________________________
WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 9). These functions will not be directly callable after loading.
shape of train input (images)      : (2000, 120, 120, 3)
shape of train input (gender prob) : (2000, 2)
shape of train output              : (2000, 1)
tf.Tensor(
[[0.8962132 ]
 [0.5019521 ]
 [0.7241342 ]
 [0.01662456]
 [0.01564574]
 [0.5961059 ]
 [0.49420092]
 [0.72408664]
 [0.01783437]
 [0.67057604]
 [0.49064162]
 [0.49672267]
 [0.49607393]
 [0.0044527 ]
 [0.81132203]], shape=(15, 1), dtype=float32)
```

맨 아래쪽의 ```tf.Tensor``` 부분의 **15개의 값 (모델 출력값) 이 서로 비슷하면, 모델 출력이 평균 등 특정 값으로 수렴** 했다는 의미이므로 **학습이 제대로 되지 않은 것** 이다.