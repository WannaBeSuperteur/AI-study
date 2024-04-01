## 모델
```classify_male_or_female```

## 모델의 목적
이미지에 있는 가상 인물의 성별을 예측한다.

## 모델 학습 로그
validation accuracy : **94.33%**
* 인간의 눈으로 성별을 명확히 구분할 수 있는 이미지가 전체의 95%라고 가정할 때, 명확히 구분할 수 없는 이미지에 대한 정확도를 50%로 가정하면 인간의 정확도는 **97.5%** 이다. 이보다 다소 낮지만 성능이 우려되는 정도는 아니다.
  * 성능 등을 고려할 때 필요한 경우, C-VAE 모델의 condition 값으로 **특정 성별이면 1, 아니면 0** 의 one-hot 값 대신 **특정 성별일 확률** (softmax 값) 을 이용할 수도 있다.

```
Epoch 1/15
194/194 [==============================] - 175s 900ms/step - loss: 0.6416 - accuracy: 0.7636 - val_loss: 0.5429 - val_accuracy: 0.7340 - lr: 0.0010
Epoch 2/15
194/194 [==============================] - 196s 1s/step - loss: 0.3867 - accuracy: 0.8687 - val_loss: 0.3352 - val_accuracy: 0.8648 - lr: 0.0010
Epoch 3/15
194/194 [==============================] - 173s 891ms/step - loss: 0.3010 - accuracy: 0.9028 - val_loss: 0.2344 - val_accuracy: 0.9331 - lr: 0.0010
Epoch 4/15
194/194 [==============================] - 208s 1s/step - loss: 0.2623 - accuracy: 0.9124 - val_loss: 0.1960 - val_accuracy: 0.9520 - lr: 0.0010
Epoch 5/15
194/194 [==============================] - 187s 963ms/step - loss: 0.2347 - accuracy: 0.9279 - val_loss: 0.2489 - val_accuracy: 0.9128 - lr: 0.0010
Epoch 6/15
194/194 [==============================] - 200s 1s/step - loss: 0.2163 - accuracy: 0.9308 - val_loss: 0.2213 - val_accuracy: 0.9317 - lr: 0.0010
Epoch 7/15
194/194 [==============================] - 202s 1s/step - loss: 0.1628 - accuracy: 0.9562 - val_loss: 0.2198 - val_accuracy: 0.9317 - lr: 1.0000e-04
Epoch 8/15
194/194 [==============================] - 200s 1s/step - loss: 0.1404 - accuracy: 0.9636 - val_loss: 0.1897 - val_accuracy: 0.9491 - lr: 1.0000e-04
Epoch 9/15
194/194 [==============================] - 202s 1s/step - loss: 0.1289 - accuracy: 0.9701 - val_loss: 0.1919 - val_accuracy: 0.9506 - lr: 1.0000e-04
Epoch 10/15
194/194 [==============================] - 199s 1s/step - loss: 0.1202 - accuracy: 0.9698 - val_loss: 0.2008 - val_accuracy: 0.9462 - lr: 1.0000e-04
Epoch 11/15
194/194 [==============================] - 206s 1s/step - loss: 0.1087 - accuracy: 0.9769 - val_loss: 0.1897 - val_accuracy: 0.9491 - lr: 1.0000e-05
Epoch 12/15
194/194 [==============================] - 183s 942ms/step - loss: 0.1018 - accuracy: 0.9791 - val_loss: 0.2116 - val_accuracy: 0.9419 - lr: 1.0000e-05
Epoch 13/15
194/194 [==============================] - 174s 895ms/step - loss: 0.1017 - accuracy: 0.9801 - val_loss: 0.2080 - val_accuracy: 0.9419 - lr: 1.0000e-06
Epoch 14/15
194/194 [==============================] - 173s 889ms/step - loss: 0.1021 - accuracy: 0.9804 - val_loss: 0.2048 - val_accuracy: 0.9433 - lr: 1.0000e-06
Epoch 15/15
194/194 [==============================] - 172s 886ms/step - loss: 0.1039 - accuracy: 0.9790 - val_loss: 0.2047 - val_accuracy: 0.9433 - lr: 1.0000e-07
Model: "classify__male__or__female_cnn__model"
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

 dense_0 (Dense)             multiple                  3965440

 dense_1 (Dense)             multiple                  32832

 dense_final (Dense)         multiple                  130

=================================================================
Total params: 4,100,898
Trainable params: 4,100,898
Non-trainable params: 0
_________________________________________________________________
```