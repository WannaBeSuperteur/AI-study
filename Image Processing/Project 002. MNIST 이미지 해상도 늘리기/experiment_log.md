# Model 1 (2024.03.09 13:11)
## 이전 모델과의 차이점
* 직전 모델에는 4개의 central pixel 에 대한 정보가 개별 output에 대한 첫번째 Dense Layer (vector size = 128) 에만 추가로 concatenate 되었음.
* Model 1 에서는 개별 output에 대한 첫번째 Dense Layer (vector size = 128) 뿐 아니라 **두번째 Dense Layer (vector size = 32) 에도 추가로 concatenate 됨.

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