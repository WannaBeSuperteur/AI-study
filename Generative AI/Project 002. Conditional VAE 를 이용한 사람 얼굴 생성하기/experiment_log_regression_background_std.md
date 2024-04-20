## 모델
```regression_background_std```

## 모델의 목적
이미지에 있는 인물 사진의 배경에 대해, 그 색의 편차 (특히 인접한 픽셀 간의 차이) 의 정도를 예측한다.
* 0부터 1까지의 값
* ```1.0``` : 편차 (차이)가 큼, ```0.5``` : 편차 (차이)가 중간임, ```0.0``` : 편차 (차이)가 거의 없이 배경 색이 일정함

## 모델 학습 로그
* 학습 설정
  * max epochs : **40 epochs**
  * 학습 데이터 개수 : 남성 1,000 장, 여성 1,000 장의 **총 2,000 장 이미지**
* 학습 결과 : 각 성별 평균값으로 예측했을 때에 비해 MSE 기준으로 오차가 **1.31배** 작음
  * validation loss (MSE) : **0.1169**
  * 각 성별에 대해 평균값 (male: 0.6275, female: 0.6630) 으로 예측했을 때의 MSE (male: 0.1650, female: 0.1409) 의 평균값 : **0.1530** 

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
[[0.5]
 [1. ]
 [0. ]
 ...
 [0.5]
 [0.5]
 [1. ]]

Epoch 1/40
57/57 [==============================] - 38s 658ms/step - loss: 0.5567 - val_loss: 0.2594 - lr: 4.8000e-04
Epoch 2/40
57/57 [==============================] - 46s 815ms/step - loss: 0.2333 - val_loss: 0.1892 - lr: 4.8000e-04
Epoch 3/40
57/57 [==============================] - 49s 863ms/step - loss: 0.1879 - val_loss: 0.1895 - lr: 4.8000e-04
Epoch 4/40
57/57 [==============================] - 49s 864ms/step - loss: 0.1710 - val_loss: 0.1524 - lr: 4.8000e-04
Epoch 5/40
57/57 [==============================] - 50s 883ms/step - loss: 0.1621 - val_loss: 0.1558 - lr: 4.8000e-04
Epoch 6/40
57/57 [==============================] - 50s 873ms/step - loss: 0.1543 - val_loss: 0.1360 - lr: 4.8000e-04
Epoch 7/40
57/57 [==============================] - 49s 861ms/step - loss: 0.1377 - val_loss: 0.1218 - lr: 4.8000e-04
Epoch 8/40
57/57 [==============================] - 49s 867ms/step - loss: 0.1272 - val_loss: 0.1657 - lr: 4.8000e-04
Epoch 9/40
57/57 [==============================] - 49s 864ms/step - loss: 0.1288 - val_loss: 0.1289 - lr: 4.8000e-04
Epoch 10/40
57/57 [==============================] - 51s 894ms/step - loss: 0.1135 - val_loss: 0.1203 - lr: 1.2000e-04
Epoch 11/40
57/57 [==============================] - 60s 1s/step - loss: 0.1098 - val_loss: 0.1221 - lr: 1.2000e-04
Epoch 12/40
57/57 [==============================] - 49s 869ms/step - loss: 0.1070 - val_loss: 0.1155 - lr: 1.2000e-04
Epoch 13/40
57/57 [==============================] - 49s 867ms/step - loss: 0.1040 - val_loss: 0.1139 - lr: 1.2000e-04
Epoch 14/40
57/57 [==============================] - 55s 967ms/step - loss: 0.1026 - val_loss: 0.1135 - lr: 1.2000e-04
Epoch 15/40
57/57 [==============================] - 55s 970ms/step - loss: 0.1009 - val_loss: 0.1163 - lr: 1.2000e-04
Epoch 16/40
57/57 [==============================] - 51s 893ms/step - loss: 0.0973 - val_loss: 0.1168 - lr: 1.2000e-04
Epoch 17/40
57/57 [==============================] - 50s 880ms/step - loss: 0.0936 - val_loss: 0.1168 - lr: 3.0000e-05
Epoch 18/40
57/57 [==============================] - 50s 871ms/step - loss: 0.0934 - val_loss: 0.1161 - lr: 3.0000e-05
Epoch 19/40
57/57 [==============================] - 49s 866ms/step - loss: 0.0903 - val_loss: 0.1173 - lr: 7.5000e-06
Epoch 20/40
57/57 [==============================] - 49s 869ms/step - loss: 0.0903 - val_loss: 0.1158 - lr: 7.5000e-06
Epoch 21/40
57/57 [==============================] - 50s 873ms/step - loss: 0.0896 - val_loss: 0.1166 - lr: 1.8750e-06
Epoch 22/40
57/57 [==============================] - 49s 865ms/step - loss: 0.0893 - val_loss: 0.1169 - lr: 1.8750e-06
Epoch 23/40
57/57 [==============================] - 60s 1s/step - loss: 0.0887 - val_loss: 0.1168 - lr: 4.6875e-07
Epoch 24/40
57/57 [==============================] - 53s 928ms/step - loss: 0.0900 - val_loss: 0.1169 - lr: 4.6875e-07
Model: "regression__background__std__model"
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

 conv2d_4 (Conv2D)           multiple                  73856

 conv2d_5 (Conv2D)           multiple                  221376

 dense_0 (Dense)             multiple                  7963136

 dense_1 (Dense)             multiple                  32960

 dense_final (Dense)         multiple                  67

=================================================================
Total params: 8,356,963
Trainable params: 8,356,963
Non-trainable params: 0
_________________________________________________________________
WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 10). These functions will not be directly callable after loading.
shape of train input (images)      : (2000, 120, 120, 3)
shape of train input (gender prob) : (2000, 2)
shape of train output              : (2000, 1)
tf.Tensor(
[[0.3654331 ]
 [0.99909496]
 [0.18406819]
 [0.83364433]
 [0.40128362]
 [0.10949367]
 [0.7654471 ]
 [0.26267985]
 [0.98847055]
 [0.4490595 ]
 [0.6279733 ]
 [0.99992585]
 [0.9056929 ]
 [0.28841683]
 [0.589686  ]], shape=(15, 1), dtype=float32)
```

맨 아래쪽의 ```tf.Tensor``` 부분의 **15개의 값 (모델 출력값) 이 서로 비슷하면, 모델 출력이 평균 등 특정 값으로 수렴** 했다는 의미이므로 **학습이 제대로 되지 않은 것** 이다.