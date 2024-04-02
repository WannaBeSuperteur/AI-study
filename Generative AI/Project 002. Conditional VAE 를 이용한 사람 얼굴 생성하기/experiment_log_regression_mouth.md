## 모델
```regression_mouth```

## 모델의 목적
이미지에 있는 인물이 입을 벌린 정도를 예측한다.
* 0부터 1까지의 값
* ```1.0``` : 입을 완전히 벌림, ```0.5``` : 입을 조금 벌림, ```0.0``` : 입을 다문 상태

## 모델 학습 로그
* 학습 설정
  * max epochs : **40 epochs**
  * 학습 데이터 개수 : 남성 1,000 장, 여성 1,000 장의 **총 2,000 장 이미지**
* 학습 결과 : 각 성별 평균값으로 예측했을 때에 비해 MSE 기준으로 오차가 **7.91배** 작음
  * validation loss (MSE) : **0.0258**
  * 각 성별에 대해 평균값 (male: 0.4370, female: 0.6195) 으로 예측했을 때의 MSE (male: 0.2060, female: 0.2020) 의 평균값 : **0.2040** 

```
Epoch 1/40
57/57 [==============================] - 9s 136ms/step - loss: 0.3502 - val_loss: 0.1121 - lr: 0.0010
Epoch 2/40
57/57 [==============================] - 8s 146ms/step - loss: 0.0996 - val_loss: 0.0618 - lr: 0.0010
Epoch 3/40
57/57 [==============================] - 9s 155ms/step - loss: 0.0655 - val_loss: 0.0502 - lr: 0.0010
Epoch 4/40
57/57 [==============================] - 9s 157ms/step - loss: 0.0527 - val_loss: 0.0378 - lr: 0.0010
Epoch 5/40
57/57 [==============================] - 10s 173ms/step - loss: 0.0440 - val_loss: 0.0362 - lr: 0.0010
Epoch 6/40
57/57 [==============================] - 11s 188ms/step - loss: 0.0406 - val_loss: 0.0369 - lr: 0.0010
Epoch 7/40
57/57 [==============================] - 11s 187ms/step - loss: 0.0396 - val_loss: 0.0370 - lr: 0.0010
Epoch 8/40
57/57 [==============================] - 12s 214ms/step - loss: 0.0331 - val_loss: 0.0303 - lr: 1.0000e-04
Epoch 9/40
57/57 [==============================] - 12s 212ms/step - loss: 0.0306 - val_loss: 0.0292 - lr: 1.0000e-04
Epoch 10/40
57/57 [==============================] - 12s 209ms/step - loss: 0.0290 - val_loss: 0.0284 - lr: 1.0000e-04
Epoch 11/40
57/57 [==============================] - 12s 217ms/step - loss: 0.0283 - val_loss: 0.0279 - lr: 1.0000e-04
Epoch 12/40
57/57 [==============================] - 12s 212ms/step - loss: 0.0278 - val_loss: 0.0275 - lr: 1.0000e-04
Epoch 13/40
57/57 [==============================] - 12s 212ms/step - loss: 0.0269 - val_loss: 0.0266 - lr: 1.0000e-04
Epoch 14/40
57/57 [==============================] - 12s 210ms/step - loss: 0.0264 - val_loss: 0.0271 - lr: 1.0000e-04
Epoch 15/40
57/57 [==============================] - 12s 211ms/step - loss: 0.0261 - val_loss: 0.0258 - lr: 1.0000e-04
Epoch 16/40
57/57 [==============================] - 13s 220ms/step - loss: 0.0247 - val_loss: 0.0260 - lr: 1.0000e-04
Epoch 17/40
57/57 [==============================] - 12s 211ms/step - loss: 0.0245 - val_loss: 0.0262 - lr: 1.0000e-04
Epoch 18/40
57/57 [==============================] - 12s 213ms/step - loss: 0.0237 - val_loss: 0.0259 - lr: 1.0000e-05
Epoch 19/40
57/57 [==============================] - 12s 214ms/step - loss: 0.0237 - val_loss: 0.0258 - lr: 1.0000e-05
Epoch 20/40
57/57 [==============================] - 12s 209ms/step - loss: 0.0234 - val_loss: 0.0258 - lr: 1.0000e-06
Epoch 21/40
57/57 [==============================] - 12s 219ms/step - loss: 0.0236 - val_loss: 0.0258 - lr: 1.0000e-06
Epoch 22/40
57/57 [==============================] - 12s 212ms/step - loss: 0.0235 - val_loss: 0.0258 - lr: 1.0000e-07
Epoch 23/40
57/57 [==============================] - 12s 214ms/step - loss: 0.0235 - val_loss: 0.0258 - lr: 1.0000e-07
Epoch 24/40
57/57 [==============================] - 12s 210ms/step - loss: 0.0238 - val_loss: 0.0258 - lr: 1.0000e-08
Epoch 25/40
57/57 [==============================] - 12s 212ms/step - loss: 0.0235 - val_loss: 0.0258 - lr: 1.0000e-08
Epoch 26/40
57/57 [==============================] - 13s 221ms/step - loss: 0.0236 - val_loss: 0.0258 - lr: 1.0000e-09
Epoch 27/40
57/57 [==============================] - 12s 217ms/step - loss: 0.0234 - val_loss: 0.0258 - lr: 1.0000e-09
Epoch 28/40
57/57 [==============================] - 12s 218ms/step - loss: 0.0234 - val_loss: 0.0258 - lr: 1.0000e-10
Epoch 29/40
57/57 [==============================] - 12s 216ms/step - loss: 0.0234 - val_loss: 0.0258 - lr: 1.0000e-10
Model: "regression__mouth__model"
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

 dense_0 (Dense)             multiple                  1327360

 dense_1 (Dense)             multiple                  16576

 dense_final (Dense)         multiple                  65

=================================================================
Total params: 1,446,497
Trainable params: 1,446,497
Non-trainable params: 0
_________________________________________________________________
WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 9). These functions will not be directly callable after loading.
shape of train input (images)      : (2000, 60, 60, 3)
shape of train input (gender prob) : (2000, 2)
shape of train output              : (2000, 1)
```