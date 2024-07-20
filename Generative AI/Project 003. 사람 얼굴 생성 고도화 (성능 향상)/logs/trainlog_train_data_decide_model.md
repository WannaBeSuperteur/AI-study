## 학습 데이터 결정 모델 테스트 로그

* 결과 요약

|data size|epochs|valid loss|valid accuracy|
|---|---|---|---|
|6,873|13|0.1735|0.9462|

```
shape of train input : (6873, 128, 104, 3)
shape of train output : (6873, 2)
loading cnn model ...
2024-07-20 15:02:43.089475: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-07-20 15:02:44.482808: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 9702 MB memory:  -> device: 0, name: Quadro M6000, pci bus id: 0000:03:00.0, compute capability: 5.2
2024-07-20 15:02:44.490443: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:1 with 9703 MB memory:  -> device: 1, name: Quadro M6000, pci bus id: 0000:22:00.0, compute capability: 5.2
train input shape : (6873, 128, 104, 3)
train output shape : (6873, 2)
Epoch 1/15
2024-07-20 15:02:48.511504: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8907
2024-07-20 15:02:49.425299: W tensorflow/stream_executor/gpu/asm_compiler.cc:111] *** WARNING *** You are using ptxas 11.0.194, which is older than 11.1. ptxas before 11.1 is known to miscompile XLA code, leading to incorrect results or invalid-address errors.

You may not need to update to CUDA 11.1; cherry-picking the ptxas binary is often sufficient.
194/194 [==============================] - 10s 34ms/step - loss: 0.6942 - accuracy: 0.6784 - val_loss: 0.3243 - val_accuracy: 0.9055 - lr: 0.0010
Epoch 2/15
194/194 [==============================] - 6s 32ms/step - loss: 0.3962 - accuracy: 0.8655 - val_loss: 0.3053 - val_accuracy: 0.9012 - lr: 0.0010
Epoch 3/15
194/194 [==============================] - 6s 31ms/step - loss: 0.2941 - accuracy: 0.9017 - val_loss: 0.1985 - val_accuracy: 0.9419 - lr: 0.0010
Epoch 4/15
194/194 [==============================] - 6s 31ms/step - loss: 0.2712 - accuracy: 0.9109 - val_loss: 0.3016 - val_accuracy: 0.8823 - lr: 0.0010
Epoch 5/15
194/194 [==============================] - 6s 32ms/step - loss: 0.2314 - accuracy: 0.9263 - val_loss: 0.2733 - val_accuracy: 0.9055 - lr: 0.0010
Epoch 6/15
194/194 [==============================] - 6s 31ms/step - loss: 0.1735 - accuracy: 0.9510 - val_loss: 0.1705 - val_accuracy: 0.9419 - lr: 1.0000e-04
Epoch 7/15
194/194 [==============================] - 6s 31ms/step - loss: 0.1536 - accuracy: 0.9567 - val_loss: 0.1927 - val_accuracy: 0.9360 - lr: 1.0000e-04
Epoch 8/15
194/194 [==============================] - 6s 31ms/step - loss: 0.1419 - accuracy: 0.9615 - val_loss: 0.1691 - val_accuracy: 0.9448 - lr: 1.0000e-04
Epoch 9/15
194/194 [==============================] - 6s 31ms/step - loss: 0.1329 - accuracy: 0.9665 - val_loss: 0.1992 - val_accuracy: 0.9360 - lr: 1.0000e-04
Epoch 10/15
194/194 [==============================] - 6s 31ms/step - loss: 0.1263 - accuracy: 0.9690 - val_loss: 0.1909 - val_accuracy: 0.9375 - lr: 1.0000e-04
Epoch 11/15
194/194 [==============================] - 6s 32ms/step - loss: 0.1117 - accuracy: 0.9774 - val_loss: 0.1748 - val_accuracy: 0.9448 - lr: 1.0000e-05
Epoch 12/15
194/194 [==============================] - 6s 32ms/step - loss: 0.1086 - accuracy: 0.9788 - val_loss: 0.1778 - val_accuracy: 0.9419 - lr: 1.0000e-05
Epoch 13/15
194/194 [==============================] - 6s 31ms/step - loss: 0.1083 - accuracy: 0.9782 - val_loss: 0.1735 - val_accuracy: 0.9462 - lr: 1.0000e-06
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

 dense_0 (Dense)             multiple                  3539456

 dense_1 (Dense)             multiple                  32832

 dense_final (Dense)         multiple                  130

=================================================================
Total params: 3,674,914
Trainable params: 3,674,914
Non-trainable params: 0
_________________________________________________________________
2024-07-20 15:04:10.232057: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
WARNING:absl:Found untraced functions such as leaky_re_lu_layer_call_fn, leaky_re_lu_layer_call_and_return_conditional_losses, leaky_re_lu_1_layer_call_fn, leaky_re_lu_1_layer_call_and_return_conditional_losses while saving (showing 4 of 4). These functions will not be directly callable after loading.
```