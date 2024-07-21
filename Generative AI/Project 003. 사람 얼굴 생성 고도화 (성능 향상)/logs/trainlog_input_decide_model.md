## 입력값 결정 모델 테스트 로그

* 학습 데이터
  * ```resize/male```, ```resize/female``` 에 있는 남녀 이미지 각각 파일 이름순 최초 1,000 장 
  * eyes, mouth 의 경우 이미지의 일부만 보고 알 수 있기 때문에 원본 대신 cropped image 를 학습 
* GAI-P2 와의 차이점
  * 성별 정보를 학습에 반영하지 않음
  * eyes, mouth 의 경우 다른 영역에 의해 모델 학습이 방해되지 않도록 눈, 입에 해당하는 영역만을 보다 정확히 추출하여 학습
  * head (고개를 돌린 정도) 추가
* GAI-P2 reference 정보
  * ```Project 002. Conditional VAE 를 이용한 사람 얼굴 생성하기/experiment_log_regression_{item}.md``` 
  * 단, ```background``` 의 경우 ```background_mean``` item 에 대한 정보를 찾으면 됨
  * valid loss N.T.
    * 모델이 제대로 학습되지 않아, 학습 후 모델 출력값이 각 성별로 동일한 성별 이미지들의 train output의 평균값으로 수렴했을 때의 valid loss
    * GAI-P2 의 위 documentation 으로부터 확인

| item       | cropped size<br>(h x w) | epochs | final valid loss | final valid loss<br>(GAI-P2) | 성능 개선율 | valid loss N.T.<br>(from GAI-P2 doc) |
|------------|-------------------------|--------|------------------|------------------------------|-----------|--------------------------------------|
| background | 104 x 128 (original)    | 23     | 0.0325           | 0.0490                       | 📈 +51%   | 0.0917                               |
| eyes       | 56 x 24                 | 23     | 0.0191           | 0.0205                       | 📈 +7%    | 0.0339                               |
| hair_color | 104 x 128 (original)    | 17     | 0.0213           | 0.0312                       | 📈 +46%   | 0.0614                               |
| head       | 104 x 128 (original)    | 17     | 0.0406           | -                            | -         |                                      |
| mouth      | 32 x 16                 |  20    | 0.0329           | 0.0258                       | 📉 -22%   | 0.2040                               |

### background 🏞

```
input type                    : background
shape of train input (images) : (2000, 128, 104, 3)
shape of train output         : (2000, 1)
train input shape : (2000, 128, 104, 3)
train output shape : (2000, 1)
Epoch 1/30
2024-07-21 10:35:55.787479: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-07-21 10:35:55.950750: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
57/57 [==============================] - 4s 39ms/step - loss: 0.2369 - accuracy: 0.3067 - val_loss: 0.1046 - val_accuracy: 0.3000 - lr: 0.0010
Epoch 2/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0895 - accuracy: 0.3600 - val_loss: 0.0771 - val_accuracy: 0.2950 - lr: 0.0010
Epoch 3/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0684 - accuracy: 0.3611 - val_loss: 0.0535 - val_accuracy: 0.3050 - lr: 0.0010
Epoch 4/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0596 - accuracy: 0.3622 - val_loss: 0.0498 - val_accuracy: 0.3150 - lr: 0.0010
Epoch 5/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0529 - accuracy: 0.3683 - val_loss: 0.0484 - val_accuracy: 0.3100 - lr: 0.0010
Epoch 6/30
57/57 [==============================] - 2s 33ms/step - loss: 0.0501 - accuracy: 0.3683 - val_loss: 0.0486 - val_accuracy: 0.3100 - lr: 0.0010
Epoch 7/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0491 - accuracy: 0.3639 - val_loss: 0.0411 - val_accuracy: 0.3050 - lr: 0.0010
Epoch 8/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0463 - accuracy: 0.3656 - val_loss: 0.0416 - val_accuracy: 0.3100 - lr: 0.0010
Epoch 9/30
57/57 [==============================] - 2s 33ms/step - loss: 0.0462 - accuracy: 0.3644 - val_loss: 0.0379 - val_accuracy: 0.3150 - lr: 0.0010
Epoch 10/30
57/57 [==============================] - 2s 33ms/step - loss: 0.0416 - accuracy: 0.3667 - val_loss: 0.0402 - val_accuracy: 0.3050 - lr: 0.0010
Epoch 11/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0427 - accuracy: 0.3667 - val_loss: 0.0511 - val_accuracy: 0.3050 - lr: 0.0010
Epoch 12/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0376 - accuracy: 0.3694 - val_loss: 0.0384 - val_accuracy: 0.3100 - lr: 1.0000e-04
Epoch 13/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0350 - accuracy: 0.3694 - val_loss: 0.0365 - val_accuracy: 0.3100 - lr: 1.0000e-04
Epoch 14/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0327 - accuracy: 0.3694 - val_loss: 0.0355 - val_accuracy: 0.3150 - lr: 1.0000e-04
Epoch 15/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0327 - accuracy: 0.3700 - val_loss: 0.0343 - val_accuracy: 0.3100 - lr: 1.0000e-04
Epoch 16/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0314 - accuracy: 0.3689 - val_loss: 0.0343 - val_accuracy: 0.3100 - lr: 1.0000e-04
Epoch 17/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0299 - accuracy: 0.3706 - val_loss: 0.0321 - val_accuracy: 0.3150 - lr: 1.0000e-04
Epoch 18/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0291 - accuracy: 0.3689 - val_loss: 0.0314 - val_accuracy: 0.3150 - lr: 1.0000e-04
Epoch 19/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0287 - accuracy: 0.3694 - val_loss: 0.0347 - val_accuracy: 0.3150 - lr: 1.0000e-04
Epoch 20/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0271 - accuracy: 0.3689 - val_loss: 0.0322 - val_accuracy: 0.3150 - lr: 1.0000e-04
Epoch 21/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0263 - accuracy: 0.3700 - val_loss: 0.0331 - val_accuracy: 0.3100 - lr: 1.0000e-05
Epoch 22/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0256 - accuracy: 0.3700 - val_loss: 0.0325 - val_accuracy: 0.3100 - lr: 1.0000e-05
Epoch 23/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0258 - accuracy: 0.3706 - val_loss: 0.0325 - val_accuracy: 0.3100 - lr: 1.0000e-06
Model: "regression__background__model"
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

 dense_0 (Dense)             multiple                  1769728

 dense_1 (Dense)             multiple                  16448

 dense_final (Dense)         multiple                  65

=================================================================
Total params: 1,888,737
Trainable params: 1,888,737
Non-trainable params: 0
_________________________________________________________________
2024-07-21 10:36:38.853760: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:absl:Found untraced functions such as leaky_re_lu_layer_call_fn, leaky_re_lu_layer_call_and_return_conditional_losses, leaky_re_lu_1_layer_call_fn, leaky_re_lu_1_layer_call_and_return_conditional_losses while saving (showing 4 of 4). These functions will not be directly callable after loading.

```

### eyes 👀

```
input type                    : eyes
shape of train input (images) : (2000, 24, 56, 3)
shape of train output         : (2000, 1)
train input shape : (2000, 24, 56, 3)
train output shape : (2000, 1)
Epoch 1/30
57/57 [==============================] - 2s 19ms/step - loss: 0.3558 - accuracy: 0.8489 - val_loss: 0.0825 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 2/30
57/57 [==============================] - 1s 14ms/step - loss: 0.0613 - accuracy: 0.8511 - val_loss: 0.0439 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 3/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0426 - accuracy: 0.8511 - val_loss: 0.0375 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 4/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0391 - accuracy: 0.8511 - val_loss: 0.0327 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 5/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0315 - accuracy: 0.8511 - val_loss: 0.0253 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 6/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0241 - accuracy: 0.8511 - val_loss: 0.0270 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 7/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0220 - accuracy: 0.8511 - val_loss: 0.0198 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 8/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0210 - accuracy: 0.8511 - val_loss: 0.0231 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 9/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0195 - accuracy: 0.8511 - val_loss: 0.0232 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 10/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0182 - accuracy: 0.8511 - val_loss: 0.0234 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 11/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0171 - accuracy: 0.8511 - val_loss: 0.0178 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 12/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0178 - accuracy: 0.8511 - val_loss: 0.0205 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 13/30
57/57 [==============================] - 1s 14ms/step - loss: 0.0160 - accuracy: 0.8511 - val_loss: 0.0195 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 14/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0159 - accuracy: 0.8511 - val_loss: 0.0186 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 15/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0151 - accuracy: 0.8517 - val_loss: 0.0220 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 16/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0152 - accuracy: 0.8511 - val_loss: 0.0254 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 17/30
57/57 [==============================] - 1s 14ms/step - loss: 0.0129 - accuracy: 0.8517 - val_loss: 0.0268 - val_accuracy: 0.8500 - lr: 0.0010
Epoch 18/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0135 - accuracy: 0.8528 - val_loss: 0.0203 - val_accuracy: 0.8600 - lr: 0.0010
Epoch 19/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0125 - accuracy: 0.8533 - val_loss: 0.0247 - val_accuracy: 0.8500 - lr: 0.0010
Epoch 20/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0119 - accuracy: 0.8544 - val_loss: 0.0208 - val_accuracy: 0.8550 - lr: 0.0010
Epoch 21/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0113 - accuracy: 0.8539 - val_loss: 0.0180 - val_accuracy: 0.8550 - lr: 0.0010
Epoch 22/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0104 - accuracy: 0.8544 - val_loss: 0.0270 - val_accuracy: 0.8350 - lr: 0.0010
Epoch 23/30
57/57 [==============================] - 1s 13ms/step - loss: 0.0093 - accuracy: 0.8550 - val_loss: 0.0191 - val_accuracy: 0.8500 - lr: 0.0010
Model: "regression__eyes__model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten_1 (Flatten)         multiple                  0

 max_pooling2d_1 (MaxPooling  multiple                 0
 2D)

 dropout (Dropout)           multiple                  0

 conv2d_5 (Conv2D)           multiple                  896

 conv2d_6 (Conv2D)           multiple                  18496

 conv2d_7 (Conv2D)           multiple                  73856

 conv2d_8 (Conv2D)           multiple                  295168

 dense_0 (Dense)             multiple                  2621952

 dense_1 (Dense)             multiple                  65664

 dense_final (Dense)         multiple                  129

=================================================================
Total params: 3,076,161
Trainable params: 3,076,161
Non-trainable params: 0
_________________________________________________________________
WARNING:absl:Found untraced functions such as leaky_re_lu_2_layer_call_fn, leaky_re_lu_2_layer_call_and_return_conditional_losses, leaky_re_lu_3_layer_call_fn, leaky_re_lu_3_layer_call_and_return_conditional_losses while saving (showing 4 of 4). These functions will not be directly callable after loading.
```

### hair_color 👩‍🦰

```
input type                    : hair_color
shape of train input (images) : (2000, 128, 104, 3)
shape of train output         : (2000, 1)
train input shape : (2000, 128, 104, 3)
train output shape : (2000, 1)
Epoch 1/30
57/57 [==============================] - 3s 35ms/step - loss: 0.2605 - accuracy: 0.6978 - val_loss: 0.0873 - val_accuracy: 0.7500 - lr: 0.0010
Epoch 2/30
57/57 [==============================] - 2s 33ms/step - loss: 0.0739 - accuracy: 0.7067 - val_loss: 0.0489 - val_accuracy: 0.7500 - lr: 0.0010
Epoch 3/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0493 - accuracy: 0.7094 - val_loss: 0.0470 - val_accuracy: 0.7500 - lr: 0.0010
Epoch 4/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0423 - accuracy: 0.7117 - val_loss: 0.0337 - val_accuracy: 0.7500 - lr: 0.0010
Epoch 5/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0400 - accuracy: 0.7128 - val_loss: 0.0260 - val_accuracy: 0.7450 - lr: 0.0010
Epoch 6/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0364 - accuracy: 0.7161 - val_loss: 0.0242 - val_accuracy: 0.7450 - lr: 0.0010
Epoch 7/30
57/57 [==============================] - 2s 30ms/step - loss: 0.0359 - accuracy: 0.7144 - val_loss: 0.0282 - val_accuracy: 0.7500 - lr: 0.0010
Epoch 8/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0339 - accuracy: 0.7178 - val_loss: 0.0221 - val_accuracy: 0.7500 - lr: 0.0010
Epoch 9/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0326 - accuracy: 0.7167 - val_loss: 0.0229 - val_accuracy: 0.7500 - lr: 0.0010
Epoch 10/30
57/57 [==============================] - 2s 33ms/step - loss: 0.0326 - accuracy: 0.7189 - val_loss: 0.0237 - val_accuracy: 0.7500 - lr: 0.0010
Epoch 11/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0293 - accuracy: 0.7183 - val_loss: 0.0215 - val_accuracy: 0.7500 - lr: 1.0000e-04
Epoch 12/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0286 - accuracy: 0.7200 - val_loss: 0.0211 - val_accuracy: 0.7500 - lr: 1.0000e-04
Epoch 13/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0283 - accuracy: 0.7194 - val_loss: 0.0213 - val_accuracy: 0.7500 - lr: 1.0000e-04
Epoch 14/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0275 - accuracy: 0.7211 - val_loss: 0.0214 - val_accuracy: 0.7500 - lr: 1.0000e-04
Epoch 15/30
57/57 [==============================] - 2s 32ms/step - loss: 0.0272 - accuracy: 0.7206 - val_loss: 0.0213 - val_accuracy: 0.7500 - lr: 1.0000e-05
Epoch 16/30
57/57 [==============================] - 2s 30ms/step - loss: 0.0273 - accuracy: 0.7211 - val_loss: 0.0213 - val_accuracy: 0.7500 - lr: 1.0000e-05
Epoch 17/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0274 - accuracy: 0.7211 - val_loss: 0.0213 - val_accuracy: 0.7500 - lr: 1.0000e-06
Model: "regression__hair__color__model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten_2 (Flatten)         multiple                  0

 max_pooling2d_2 (MaxPooling  multiple                 0
 2D)

 dropout (Dropout)           multiple                  0

 conv2d_9 (Conv2D)           multiple                  896

 conv2d_10 (Conv2D)          multiple                  9248

 conv2d_11 (Conv2D)          multiple                  18496

 conv2d_12 (Conv2D)          multiple                  36928

 conv2d_13 (Conv2D)          multiple                  36928

 dense_0 (Dense)             multiple                  1769728

 dense_1 (Dense)             multiple                  16448

 dense_final (Dense)         multiple                  65

=================================================================
Total params: 1,888,737
Trainable params: 1,888,737
Non-trainable params: 0
_________________________________________________________________
WARNING:absl:Found untraced functions such as leaky_re_lu_4_layer_call_fn, leaky_re_lu_4_layer_call_and_return_conditional_losses, leaky_re_lu_5_layer_call_fn, leaky_re_lu_5_layer_call_and_return_conditional_losses while saving (showing 4 of 4). These functions will not be directly callable after loading.

```

### head 🐴

```
input type                    : head
shape of train input (images) : (2000, 128, 104, 3)
shape of train output         : (2000, 1)
train input shape : (2000, 128, 104, 3)
train output shape : (2000, 1)
Epoch 1/30
57/57 [==============================] - 3s 36ms/step - loss: 0.2173 - accuracy: 0.2044 - val_loss: 0.0938 - val_accuracy: 0.2350 - lr: 0.0010
Epoch 2/30
57/57 [==============================] - 2s 33ms/step - loss: 0.0590 - accuracy: 0.2606 - val_loss: 0.0637 - val_accuracy: 0.2350 - lr: 0.0010
Epoch 3/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0392 - accuracy: 0.2611 - val_loss: 0.0581 - val_accuracy: 0.2350 - lr: 0.0010
Epoch 4/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0353 - accuracy: 0.2578 - val_loss: 0.0578 - val_accuracy: 0.2350 - lr: 0.0010
Epoch 5/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0298 - accuracy: 0.2611 - val_loss: 0.0420 - val_accuracy: 0.2350 - lr: 0.0010
Epoch 6/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0257 - accuracy: 0.2617 - val_loss: 0.0470 - val_accuracy: 0.2300 - lr: 0.0010
Epoch 7/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0247 - accuracy: 0.2622 - val_loss: 0.0467 - val_accuracy: 0.2250 - lr: 0.0010
Epoch 8/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0225 - accuracy: 0.2622 - val_loss: 0.0419 - val_accuracy: 0.2350 - lr: 1.0000e-04
Epoch 9/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0202 - accuracy: 0.2617 - val_loss: 0.0408 - val_accuracy: 0.2350 - lr: 1.0000e-04
Epoch 10/30
57/57 [==============================] - 2s 30ms/step - loss: 0.0196 - accuracy: 0.2617 - val_loss: 0.0408 - val_accuracy: 0.2350 - lr: 1.0000e-04
Epoch 11/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0188 - accuracy: 0.2611 - val_loss: 0.0405 - val_accuracy: 0.2350 - lr: 1.0000e-04
Epoch 12/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0185 - accuracy: 0.2617 - val_loss: 0.0402 - val_accuracy: 0.2350 - lr: 1.0000e-04
Epoch 13/30
57/57 [==============================] - 2s 30ms/step - loss: 0.0184 - accuracy: 0.2622 - val_loss: 0.0405 - val_accuracy: 0.2350 - lr: 1.0000e-04
Epoch 14/30
57/57 [==============================] - 2s 30ms/step - loss: 0.0178 - accuracy: 0.2628 - val_loss: 0.0407 - val_accuracy: 0.2350 - lr: 1.0000e-04
Epoch 15/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0169 - accuracy: 0.2622 - val_loss: 0.0406 - val_accuracy: 0.2350 - lr: 1.0000e-05
Epoch 16/30
57/57 [==============================] - 2s 30ms/step - loss: 0.0169 - accuracy: 0.2622 - val_loss: 0.0407 - val_accuracy: 0.2350 - lr: 1.0000e-05
Epoch 17/30
57/57 [==============================] - 2s 31ms/step - loss: 0.0165 - accuracy: 0.2622 - val_loss: 0.0406 - val_accuracy: 0.2350 - lr: 1.0000e-06
Model: "regression__head__model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten_3 (Flatten)         multiple                  0

 max_pooling2d_3 (MaxPooling  multiple                 0
 2D)

 dropout (Dropout)           multiple                  0

 conv2d_14 (Conv2D)          multiple                  896

 conv2d_15 (Conv2D)          multiple                  9248

 conv2d_16 (Conv2D)          multiple                  18496

 conv2d_17 (Conv2D)          multiple                  36928

 conv2d_18 (Conv2D)          multiple                  36928

 dense_0 (Dense)             multiple                  1769728

 dense_1 (Dense)             multiple                  16448

 dense_final (Dense)         multiple                  65

=================================================================
Total params: 1,888,737
Trainable params: 1,888,737
Non-trainable params: 0
_________________________________________________________________
WARNING:absl:Found untraced functions such as leaky_re_lu_6_layer_call_fn, leaky_re_lu_6_layer_call_and_return_conditional_losses, leaky_re_lu_7_layer_call_fn, leaky_re_lu_7_layer_call_and_return_conditional_losses while saving (showing 4 of 4). These functions will not be directly callable after loading.

```

### mouth 👄

```
input type                    : mouth
shape of train input (images) : (2000, 16, 32, 3)
shape of train output         : (2000, 1)
train input shape : (2000, 16, 32, 3)
train output shape : (2000, 1)
Epoch 1/30
57/57 [==============================] - 2s 20ms/step - loss: 0.4686 - accuracy: 0.6606 - val_loss: 0.1554 - val_accuracy: 0.7300 - lr: 0.0010
Epoch 2/30
57/57 [==============================] - 1s 12ms/step - loss: 0.1105 - accuracy: 0.8061 - val_loss: 0.0857 - val_accuracy: 0.7750 - lr: 0.0010
Epoch 3/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0677 - accuracy: 0.8350 - val_loss: 0.0606 - val_accuracy: 0.7850 - lr: 0.0010
Epoch 4/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0523 - accuracy: 0.8406 - val_loss: 0.0539 - val_accuracy: 0.7900 - lr: 0.0010
Epoch 5/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0415 - accuracy: 0.8472 - val_loss: 0.0421 - val_accuracy: 0.7900 - lr: 0.0010
Epoch 6/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0354 - accuracy: 0.8500 - val_loss: 0.0515 - val_accuracy: 0.7850 - lr: 0.0010
Epoch 7/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0309 - accuracy: 0.8517 - val_loss: 0.0410 - val_accuracy: 0.7800 - lr: 0.0010
Epoch 8/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0295 - accuracy: 0.8528 - val_loss: 0.0443 - val_accuracy: 0.7900 - lr: 0.0010
Epoch 9/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0278 - accuracy: 0.8506 - val_loss: 0.0466 - val_accuracy: 0.7750 - lr: 0.0010
Epoch 10/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0252 - accuracy: 0.8544 - val_loss: 0.0370 - val_accuracy: 0.7900 - lr: 1.0000e-04
Epoch 11/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0222 - accuracy: 0.8556 - val_loss: 0.0352 - val_accuracy: 0.7900 - lr: 1.0000e-04
Epoch 12/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0211 - accuracy: 0.8556 - val_loss: 0.0352 - val_accuracy: 0.7900 - lr: 1.0000e-04
Epoch 13/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0208 - accuracy: 0.8561 - val_loss: 0.0343 - val_accuracy: 0.7900 - lr: 1.0000e-04
Epoch 14/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0204 - accuracy: 0.8561 - val_loss: 0.0332 - val_accuracy: 0.7900 - lr: 1.0000e-04
Epoch 15/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0197 - accuracy: 0.8550 - val_loss: 0.0325 - val_accuracy: 0.7900 - lr: 1.0000e-04
Epoch 16/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0194 - accuracy: 0.8561 - val_loss: 0.0325 - val_accuracy: 0.7850 - lr: 1.0000e-04
Epoch 17/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0195 - accuracy: 0.8561 - val_loss: 0.0338 - val_accuracy: 0.7900 - lr: 1.0000e-04
Epoch 18/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0187 - accuracy: 0.8567 - val_loss: 0.0332 - val_accuracy: 0.7900 - lr: 1.0000e-05
Epoch 19/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0186 - accuracy: 0.8561 - val_loss: 0.0329 - val_accuracy: 0.7900 - lr: 1.0000e-05
Epoch 20/30
57/57 [==============================] - 1s 12ms/step - loss: 0.0182 - accuracy: 0.8561 - val_loss: 0.0329 - val_accuracy: 0.7900 - lr: 1.0000e-06
Model: "regression__mouth__model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten_4 (Flatten)         multiple                  0

 max_pooling2d_4 (MaxPooling  multiple                 0
 2D)

 dropout (Dropout)           multiple                  0

 conv2d_19 (Conv2D)          multiple                  896

 conv2d_20 (Conv2D)          multiple                  18496

 conv2d_21 (Conv2D)          multiple                  73856

 conv2d_22 (Conv2D)          multiple                  295168

 dense_0 (Dense)             multiple                  2621952

 dense_1 (Dense)             multiple                  65664

 dense_final (Dense)         multiple                  129

=================================================================
Total params: 3,076,161
Trainable params: 3,076,161
Non-trainable params: 0
_________________________________________________________________
WARNING:absl:Found untraced functions such as leaky_re_lu_8_layer_call_fn, leaky_re_lu_8_layer_call_and_return_conditional_losses, leaky_re_lu_9_layer_call_fn, leaky_re_lu_9_layer_call_and_return_conditional_losses while saving (showing 4 of 4). These functions will not be directly callable after loading.
```