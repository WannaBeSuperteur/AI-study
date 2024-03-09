# Model 2 (2024.03.09 14:05)
## 이전 모델과의 차이점
* 직전 모델인 Model 1 에서는 개별 output 에 대해 2개의 Dense Layer (vector size 각각 128, 32) 를 적용했으나, Model 2 에서는 3개의 Dense Layer (vector size 각각 256, 64, 16) 을 적용
* 모든 Dense Layer 에 4개의 central pixel 정보를 추가로 concatenate 하는 것은 동일
  * 따라서 개별 output 각각에 대해 central pixel 정보가 concatenate 되는 횟수가 2회 -> 3회로 증가

## 학습 로그
* total epochs = **5**

```
Epoch 1/5
6750/6750 [==============================] - 427s 63ms/step - loss: 0.0518 - val_loss: 0.0253 - lr: 0.0010
Epoch 2/5
6750/6750 [==============================] - 471s 70ms/step - loss: 0.0275 - val_loss: 0.0245 - lr: 0.0010
Epoch 3/5
6750/6750 [==============================] - 474s 70ms/step - loss: 0.0250 - val_loss: 0.0232 - lr: 0.0010
Epoch 4/5
6750/6750 [==============================] - 470s 70ms/step - loss: 0.0236 - val_loss: 0.0207 - lr: 0.0010
Epoch 5/5
6750/6750 [==============================] - 485s 72ms/step - loss: 0.0227 - val_loss: 0.0203 - lr: 0.0010
```

## 테스트 결과 및 총평
![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/47fdd7f6-d266-4fe8-89f5-f63ba0e47254)

* 이전 모델에 비해 큰 차이가 없다.
* **TODO :** 네트워크가 1단계 깊어져서 Vanishing Gradient 가 발생할 가능성이 조금 높아졌을 수 있으므로, **각 dense layer에 Batch Normalization 적용** 하여 재시도해서 Model 3을 만든다.

# Model 1 (2024.03.09 13:11)
## 이전 모델과의 차이점
* 직전 모델에는 4개의 central pixel 에 대한 정보가 개별 output에 대한 첫번째 Dense Layer (vector size = 128) 에만 추가로 concatenate 되었음.
* Model 1 에서는 개별 output에 대한 첫번째 Dense Layer (vector size = 128) 뿐 아니라 **두번째 Dense Layer (vector size = 32) 에도 추가로 concatenate** 됨.

## 학습 로그
* total epochs = **5**

```
Epoch 1/5
6750/6750 [==============================] - 404s 59ms/step - loss: 0.0398 - val_loss: 0.0242 - lr: 0.0010
Epoch 2/5
6750/6750 [==============================] - 442s 65ms/step - loss: 0.0237 - val_loss: 0.0211 - lr: 0.0010
Epoch 3/5
6750/6750 [==============================] - 399s 59ms/step - loss: 0.0218 - val_loss: 0.0200 - lr: 0.0010
Epoch 4/5
6750/6750 [==============================] - 403s 60ms/step - loss: 0.0205 - val_loss: 0.0182 - lr: 0.0010
Epoch 5/5
6750/6750 [==============================] - 412s 61ms/step - loss: 0.0195 - val_loss: 0.0186 - lr: 0.0010
```

## 테스트 결과 및 총평
![image](https://github.com/WannaBeSuperteur/AI-study/assets/32893014/47fd99ff-6881-401d-9c42-52148e315888)

* 이전 모델에 비해 다소 나아진 듯 하지만, 글씨가 있는 부분과 배경의 경계에 어색한 부분이 남아 있다.