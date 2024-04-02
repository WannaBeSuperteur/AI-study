## 모델
```regression_hair_color```

## 모델의 목적
이미지에 있는 머리 색의 밝기를 예측한다.
* 0부터 1까지의 값
* ```1.0``` : 어두운 색, ```0.5``` : 중간 색 (갈색, 회색 등), ```0.0``` : 아주 밝은 색

## 모델 학습 로그
* 학습 설정
  * max epochs : **40 epochs**
  * 학습 데이터 개수 : 남성 1,000 장, 여성 1,000 장의 **총 2,000 장 이미지**
* 학습 결과 : 평균값으로 예측했을 때에 비해 오차가 **1.97** 배 작음
  * validation loss (MSE) : **0.0312**
  * 모든 학습 데이터의 평균값 (0.8470) 으로 모든 output 값을 예측했을 때의 MSE : **0.0616**

```
Epoch 1/40
57/57 [==============================] - 38s 650ms/step - loss: 0.2342 - val_loss: 0.0810 - lr: 0.0010
Epoch 2/40
57/57 [==============================] - 52s 916ms/step - loss: 0.0666 - val_loss: 0.0450 - lr: 0.0010
Epoch 3/40
57/57 [==============================] - 46s 803ms/step - loss: 0.0485 - val_loss: 0.0407 - lr: 0.0010
Epoch 4/40
57/57 [==============================] - 46s 799ms/step - loss: 0.0394 - val_loss: 0.0333 - lr: 0.0010
Epoch 5/40
57/57 [==============================] - 46s 809ms/step - loss: 0.0388 - val_loss: 0.0316 - lr: 0.0010
Epoch 6/40
57/57 [==============================] - 57s 995ms/step - loss: 0.0357 - val_loss: 0.0320 - lr: 0.0010
Epoch 7/40
57/57 [==============================] - 57s 1s/step - loss: 0.0338 - val_loss: 0.0454 - lr: 0.0010
Epoch 8/40
57/57 [==============================] - 50s 875ms/step - loss: 0.0325 - val_loss: 0.0321 - lr: 1.0000e-04
Epoch 9/40
57/57 [==============================] - 45s 796ms/step - loss: 0.0298 - val_loss: 0.0308 - lr: 1.0000e-04
Epoch 10/40
57/57 [==============================] - 45s 796ms/step - loss: 0.0288 - val_loss: 0.0312 - lr: 1.0000e-04
Epoch 11/40
57/57 [==============================] - 56s 983ms/step - loss: 0.0285 - val_loss: 0.0326 - lr: 1.0000e-04
Epoch 12/40
57/57 [==============================] - 52s 921ms/step - loss: 0.0282 - val_loss: 0.0311 - lr: 1.0000e-05
Epoch 13/40
57/57 [==============================] - 49s 854ms/step - loss: 0.0279 - val_loss: 0.0311 - lr: 1.0000e-05
Epoch 14/40
57/57 [==============================] - 49s 854ms/step - loss: 0.0277 - val_loss: 0.0312 - lr: 1.0000e-06
Model: "regression__hair__color__model"
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
```