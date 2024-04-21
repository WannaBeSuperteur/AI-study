## 모델
```regression_eyes```

## 모델의 목적
이미지에 있는 인물이 눈을 떴는지 예측한다.
* 0부터 1까지의 값
* ```1.0``` : 눈을 완전히 뜸, ```0.5``` : 눈을 반쯤 뜨는 등 완전히 뜨지는 않음, ```0.0``` : 눈을 감았거나 거의 감음

## 모델 학습 로그
* 학습 설정
  * max epochs : **40 epochs**
  * 학습 데이터 개수 : 남성 1,000 장, 여성 1,000 장의 **총 2,000 장 이미지**
* 학습 결과 : 각 성별에 대해 해당 성별의 output 값의 평균값으로 예측했을 때에 비해 오차가 **1.65배** 작음
  * validation loss (MSE) : **0.0205**
  * 각 성별에 대해 평균값 (male: 0.9255, female: 0.9230) 으로 예측했을 때의 MSE (male: 0.0342, female: 0.0336) : 의 평균값 : **0.0339** 
  * 맨 아래쪽에 있는 size 15의 ```tf.tensor``` 는 **train 데이터 최초 15개** 에 대한 학습된 모델의 output을 출력하는 부분임
    * 성능 정량 평가가 아닌, **모델의 output이 그 평균값으로 수렴하지는 않았는지 확인** 하는 목적

```
Epoch 1/40
57/57 [==============================] - 5s 74ms/step - loss: 0.3306 - val_loss: 0.1124 - lr: 0.0010
Epoch 2/40
57/57 [==============================] - 5s 80ms/step - loss: 0.0767 - val_loss: 0.0541 - lr: 0.0010
Epoch 3/40
57/57 [==============================] - 5s 86ms/step - loss: 0.0498 - val_loss: 0.0443 - lr: 0.0010
Epoch 4/40
57/57 [==============================] - 5s 88ms/step - loss: 0.0450 - val_loss: 0.0424 - lr: 0.0010
Epoch 5/40
57/57 [==============================] - 5s 91ms/step - loss: 0.0424 - val_loss: 0.0407 - lr: 0.0010
Epoch 6/40
57/57 [==============================] - 5s 91ms/step - loss: 0.0413 - val_loss: 0.0400 - lr: 0.0010
Epoch 7/40
57/57 [==============================] - 5s 92ms/step - loss: 0.0408 - val_loss: 0.0397 - lr: 0.0010
Epoch 8/40
57/57 [==============================] - 5s 94ms/step - loss: 0.0409 - val_loss: 0.0399 - lr: 0.0010
Epoch 9/40
57/57 [==============================] - 6s 97ms/step - loss: 0.0407 - val_loss: 0.0403 - lr: 0.0010
Epoch 10/40
57/57 [==============================] - 6s 105ms/step - loss: 0.0407 - val_loss: 0.0395 - lr: 0.0010
Epoch 11/40
57/57 [==============================] - 6s 107ms/step - loss: 0.0402 - val_loss: 0.0410 - lr: 0.0010
Epoch 12/40
57/57 [==============================] - 7s 121ms/step - loss: 0.0413 - val_loss: 0.0396 - lr: 0.0010
Epoch 13/40
57/57 [==============================] - 7s 125ms/step - loss: 0.0404 - val_loss: 0.0393 - lr: 0.0010
Epoch 14/40
57/57 [==============================] - 7s 126ms/step - loss: 0.0399 - val_loss: 0.0390 - lr: 0.0010
Epoch 15/40
57/57 [==============================] - 7s 124ms/step - loss: 0.0396 - val_loss: 0.0387 - lr: 0.0010
Epoch 16/40
57/57 [==============================] - 7s 124ms/step - loss: 0.0396 - val_loss: 0.0383 - lr: 0.0010
Epoch 17/40
57/57 [==============================] - 7s 124ms/step - loss: 0.0377 - val_loss: 0.0355 - lr: 0.0010
Epoch 18/40
57/57 [==============================] - 7s 123ms/step - loss: 0.0390 - val_loss: 0.0349 - lr: 0.0010
Epoch 19/40
57/57 [==============================] - 7s 125ms/step - loss: 0.0377 - val_loss: 0.0334 - lr: 0.0010
Epoch 20/40
57/57 [==============================] - 7s 123ms/step - loss: 0.0388 - val_loss: 0.0355 - lr: 0.0010
Epoch 21/40
57/57 [==============================] - 7s 126ms/step - loss: 0.0361 - val_loss: 0.0291 - lr: 0.0010
Epoch 22/40
57/57 [==============================] - 7s 125ms/step - loss: 0.0322 - val_loss: 0.0324 - lr: 0.0010
Epoch 23/40
57/57 [==============================] - 7s 129ms/step - loss: 0.0269 - val_loss: 0.0264 - lr: 0.0010
Epoch 24/40
57/57 [==============================] - 8s 134ms/step - loss: 0.0247 - val_loss: 0.0247 - lr: 0.0010
Epoch 25/40
57/57 [==============================] - 7s 125ms/step - loss: 0.0234 - val_loss: 0.0192 - lr: 0.0010
Epoch 26/40
57/57 [==============================] - 7s 126ms/step - loss: 0.0223 - val_loss: 0.0201 - lr: 0.0010
Epoch 27/40
57/57 [==============================] - 7s 129ms/step - loss: 0.0207 - val_loss: 0.0238 - lr: 0.0010
Epoch 28/40
57/57 [==============================] - 7s 127ms/step - loss: 0.0205 - val_loss: 0.0186 - lr: 0.0010
Epoch 29/40
57/57 [==============================] - 7s 128ms/step - loss: 0.0192 - val_loss: 0.0189 - lr: 0.0010
Epoch 30/40
57/57 [==============================] - 7s 131ms/step - loss: 0.0183 - val_loss: 0.0189 - lr: 0.0010
Epoch 31/40
57/57 [==============================] - 8s 136ms/step - loss: 0.0175 - val_loss: 0.0269 - lr: 0.0010
Epoch 32/40
57/57 [==============================] - 8s 135ms/step - loss: 0.0158 - val_loss: 0.0216 - lr: 1.0000e-04
Epoch 33/40
57/57 [==============================] - 7s 128ms/step - loss: 0.0141 - val_loss: 0.0214 - lr: 1.0000e-04
Epoch 34/40
57/57 [==============================] - 7s 127ms/step - loss: 0.0138 - val_loss: 0.0204 - lr: 1.0000e-04
Epoch 35/40
57/57 [==============================] - 7s 127ms/step - loss: 0.0137 - val_loss: 0.0205 - lr: 1.0000e-05
Model: "regression__eyes__model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           multiple                  0

 max_pooling2d (MaxPooling2D  multiple                 0
 )

 dropout (Dropout)           multiple                  0

 conv2d (Conv2D)             multiple                  896

 conv2d_1 (Conv2D)           multiple                  13872

 conv2d_2 (Conv2D)           multiple                  27712

 conv2d_3 (Conv2D)           multiple                  55392

 conv2d_4 (Conv2D)           multiple                  110720

 dense_0 (Dense)             multiple                  737600

 dense_1 (Dense)             multiple                  38760

 dense_final (Dense)         multiple                  121

=================================================================
Total params: 985,073
Trainable params: 985,073
Non-trainable params: 0
_________________________________________________________________
WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 9). These functions will not be directly callable after loading.
shape of train input (images)      : (2000, 32, 60, 3)
shape of train input (gender prob) : (2000, 2)
shape of train output              : (2000, 1)
tf.Tensor(
[[0.93657076]
 [0.5333224 ]
 [0.99658227]
 [0.99982363]
 [0.9999985 ]
 [0.9980007 ]
 [0.9999778 ]
 [0.5750146 ]
 [0.70966417]
 [0.99999696]
 [0.84886503]
 [0.9839032 ]
 [0.6970676 ]
 [0.99311906]
 [0.9993832 ]], shape=(15, 1), dtype=float32)
```