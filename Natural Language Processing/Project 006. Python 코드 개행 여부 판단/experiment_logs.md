## Model 2 (2024.03.20 21:51 KST)
* 직전 모델인 **Model 1** 과의 차이점
  * 각 line (empty new line이 필요한지 판단할 부분의 직전 2개 + 직후 2개) 당 input token의 개수를 10개에서 14개로 증가

* 학습 로그 **(최종 valid loss : 0.0728)**

```
Epoch 1/80
848/848 [==============================] - 58s 57ms/step - loss: 0.1209 - val_loss: 0.0940 - lr: 0.0010
Epoch 2/80
848/848 [==============================] - 70s 82ms/step - loss: 0.0955 - val_loss: 0.0871 - lr: 0.0010
Epoch 3/80
848/848 [==============================] - 70s 83ms/step - loss: 0.0871 - val_loss: 0.0839 - lr: 0.0010
Epoch 4/80
848/848 [==============================] - 61s 72ms/step - loss: 0.0822 - val_loss: 0.0868 - lr: 0.0010
Epoch 5/80
848/848 [==============================] - 61s 71ms/step - loss: 0.0773 - val_loss: 0.0808 - lr: 0.0010
Epoch 6/80
848/848 [==============================] - 60s 71ms/step - loss: 0.0735 - val_loss: 0.0764 - lr: 0.0010
Epoch 7/80
848/848 [==============================] - 62s 73ms/step - loss: 0.0699 - val_loss: 0.0858 - lr: 0.0010
Epoch 8/80
848/848 [==============================] - 81s 95ms/step - loss: 0.0676 - val_loss: 0.0741 - lr: 0.0010
Epoch 9/80
848/848 [==============================] - 84s 99ms/step - loss: 0.0648 - val_loss: 0.0760 - lr: 0.0010
Epoch 10/80
848/848 [==============================] - 83s 98ms/step - loss: 0.0635 - val_loss: 0.0761 - lr: 0.0010
Epoch 11/80
848/848 [==============================] - 69s 81ms/step - loss: 0.0552 - val_loss: 0.0732 - lr: 1.2500e-04
Epoch 12/80
848/848 [==============================] - 76s 89ms/step - loss: 0.0526 - val_loss: 0.0729 - lr: 1.2500e-04
Epoch 13/80
848/848 [==============================] - 74s 88ms/step - loss: 0.0520 - val_loss: 0.0730 - lr: 1.2500e-04
Epoch 14/80
848/848 [==============================] - 75s 88ms/step - loss: 0.0511 - val_loss: 0.0734 - lr: 1.2500e-04
Epoch 15/80
848/848 [==============================] - 60s 71ms/step - loss: 0.0498 - val_loss: 0.0729 - lr: 1.5625e-05
Epoch 16/80
848/848 [==============================] - 68s 80ms/step - loss: 0.0496 - val_loss: 0.0727 - lr: 1.5625e-05
Epoch 17/80
848/848 [==============================] - 82s 96ms/step - loss: 0.0488 - val_loss: 0.0726 - lr: 1.5625e-05
Epoch 18/80
848/848 [==============================] - 72s 85ms/step - loss: 0.0488 - val_loss: 0.0729 - lr: 1.5625e-05
Epoch 19/80
848/848 [==============================] - 67s 79ms/step - loss: 0.0489 - val_loss: 0.0728 - lr: 1.9531e-06
Epoch 20/80
848/848 [==============================] - 75s 88ms/step - loss: 0.0486 - val_loss: 0.0728 - lr: 1.9531e-06
Epoch 21/80
848/848 [==============================] - 89s 105ms/step - loss: 0.0486 - val_loss: 0.0728 - lr: 2.4414e-07
Epoch 22/80
848/848 [==============================] - 68s 81ms/step - loss: 0.0486 - val_loss: 0.0728 - lr: 2.4414e-07
Model: "main_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dropout (Dropout)           multiple                  0

 embedding (Embedding)       multiple                  27328

 bidirectional (Bidirectiona  multiple                 49664
 l)

 bidirectional_1 (Bidirectio  multiple                 164864
 nal)

 bidirectional_2 (Bidirectio  multiple                 164864
 nal)

 bidirectional_3 (Bidirectio  multiple                 49664
 nal)

 dense (Dense)               multiple                  131136

 dense_1 (Dense)             multiple                  524416

 dense_2 (Dense)             multiple                  524416

 dense_3 (Dense)             multiple                  131136

 dense_4 (Dense)             multiple                  49280

 dense_5 (Dense)             multiple                  4128

 dense_6 (Dense)             multiple                  33

 flatten (Flatten)           multiple                  0

=================================================================
Total params: 1,820,929
Trainable params: 1,820,929
Non-trainable params: 0
_________________________________________________________________
```

* 정량 평가 결과

```
True  Positive : 1027
True  Negative : 2072
False Positive : 68
False Negative : 223

accuracy  : 0.9141592920353983
recall    : 0.8216
precision : 0.9378995433789954
F1 score  : 0.8759061833688699
```

* 정성 평가 결과
  * 직전 모델인 **Model 1** 과 동일

```
 ==== original code ====
    1 | import math
    2 | def is_prime(x):
    3 |         if x == 1 or x == 2:
    4 |                 return True
    5 |         sqrt_x = int(math.sqrt(x))
    6 |         for i in range(2, sqrt_x + 1):
    7 |                 if x % i == 0:
    8 |                         return False
    9 |         return True
   10 | for i in range(1, 30):
   11 |         print(is_prime(i))

 ==== test ====
Empty new line should be between line 2 and line 3 with prob 0.990673.
Empty new line between line 3 and line 4 with prob 0.007411.
Empty new line should be between line 4 and line 5 with prob 0.975761.
Empty new line between line 5 and line 6 with prob 0.026946.
Empty new line should be between line 6 and line 7 with prob 0.993383.
Empty new line between line 7 and line 8 with prob 0.006433.
Empty new line should be between line 8 and line 9 with prob 0.972716.
Empty new line should be between line 9 and line 10 with prob 0.502703.

 ==== AI converted code ====

import math
def is_prime(x):

        if x == 1 or x == 2:
                return True

        sqrt_x = int(math.sqrt(x))
        for i in range(2, sqrt_x + 1):

                if x % i == 0:
                        return False

        return True

for i in range(1, 30):
        print(is_prime(i))
```

## Model 1 (2024.03.20 21:06 KST)
* 학습 로그 **(최종 valid loss : 0.0792)**

```
Epoch 1/80
848/848 [==============================] - 49s 42ms/step - loss: 0.1213 - val_loss: 0.1023 - lr: 0.0010
Epoch 2/80
848/848 [==============================] - 36s 42ms/step - loss: 0.0963 - val_loss: 0.0945 - lr: 0.0010
Epoch 3/80
848/848 [==============================] - 38s 45ms/step - loss: 0.0882 - val_loss: 0.0895 - lr: 0.0010
Epoch 4/80
848/848 [==============================] - 46s 55ms/step - loss: 0.0829 - val_loss: 0.0862 - lr: 0.0010
Epoch 5/80
848/848 [==============================] - 43s 51ms/step - loss: 0.0775 - val_loss: 0.0912 - lr: 0.0010
Epoch 6/80
848/848 [==============================] - 46s 54ms/step - loss: 0.0735 - val_loss: 0.0833 - lr: 0.0010
Epoch 7/80
848/848 [==============================] - 50s 59ms/step - loss: 0.0710 - val_loss: 0.0854 - lr: 0.0010
Epoch 8/80
848/848 [==============================] - 48s 56ms/step - loss: 0.0674 - val_loss: 0.0831 - lr: 0.0010
Epoch 9/80
848/848 [==============================] - 45s 54ms/step - loss: 0.0647 - val_loss: 0.0838 - lr: 0.0010
Epoch 10/80
848/848 [==============================] - 56s 66ms/step - loss: 0.0630 - val_loss: 0.0840 - lr: 0.0010
Epoch 11/80
848/848 [==============================] - 49s 57ms/step - loss: 0.0567 - val_loss: 0.0811 - lr: 1.2500e-04
Epoch 12/80
848/848 [==============================] - 44s 52ms/step - loss: 0.0543 - val_loss: 0.0818 - lr: 1.2500e-04
Epoch 13/80
848/848 [==============================] - 59s 69ms/step - loss: 0.0524 - val_loss: 0.0808 - lr: 1.2500e-04
Epoch 14/80
848/848 [==============================] - 49s 58ms/step - loss: 0.0519 - val_loss: 0.0816 - lr: 1.2500e-04
Epoch 15/80
848/848 [==============================] - 51s 60ms/step - loss: 0.0512 - val_loss: 0.0799 - lr: 1.2500e-04
Epoch 16/80
848/848 [==============================] - 52s 61ms/step - loss: 0.0508 - val_loss: 0.0797 - lr: 1.2500e-04
Epoch 17/80
848/848 [==============================] - 57s 67ms/step - loss: 0.0491 - val_loss: 0.0799 - lr: 1.2500e-04
Epoch 18/80
848/848 [==============================] - 55s 65ms/step - loss: 0.0491 - val_loss: 0.0796 - lr: 1.2500e-04
Epoch 19/80
848/848 [==============================] - 58s 68ms/step - loss: 0.0473 - val_loss: 0.0796 - lr: 1.5625e-05
Epoch 20/80
848/848 [==============================] - 63s 75ms/step - loss: 0.0481 - val_loss: 0.0793 - lr: 1.5625e-05
Epoch 21/80
848/848 [==============================] - 51s 60ms/step - loss: 0.0482 - val_loss: 0.0791 - lr: 1.5625e-05
Epoch 22/80
848/848 [==============================] - 56s 66ms/step - loss: 0.0473 - val_loss: 0.0792 - lr: 1.5625e-05
Epoch 23/80
848/848 [==============================] - 57s 68ms/step - loss: 0.0474 - val_loss: 0.0792 - lr: 1.5625e-05
Epoch 24/80
848/848 [==============================] - 57s 67ms/step - loss: 0.0479 - val_loss: 0.0792 - lr: 1.9531e-06
Epoch 25/80
848/848 [==============================] - 57s 67ms/step - loss: 0.0469 - val_loss: 0.0792 - lr: 1.9531e-06
Epoch 26/80
848/848 [==============================] - 56s 67ms/step - loss: 0.0471 - val_loss: 0.0792 - lr: 2.4414e-07
Model: "main_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dropout (Dropout)           multiple                  0

 embedding (Embedding)       multiple                  26912

 bidirectional (Bidirectiona  multiple                 49664
 l)

 bidirectional_1 (Bidirectio  multiple                 164864
 nal)

 bidirectional_2 (Bidirectio  multiple                 164864
 nal)

 bidirectional_3 (Bidirectio  multiple                 49664
 nal)

 dense (Dense)               multiple                  98368

 dense_1 (Dense)             multiple                  393344

 dense_2 (Dense)             multiple                  393344

 dense_3 (Dense)             multiple                  98368

 dense_4 (Dense)             multiple                  49280

 dense_5 (Dense)             multiple                  4128

 dense_6 (Dense)             multiple                  33

 flatten (Flatten)           multiple                  0

=================================================================
Total params: 1,492,833
Trainable params: 1,492,833
Non-trainable params: 0
_________________________________________________________________
```

* 정량 평가 결과

```
True  Positive : 1065
True  Negative : 2068
False Positive : 67
False Negative : 190

accuracy  : 0.924188790560472
recall    : 0.848605577689243
precision : 0.9408127208480566
F1 score  : 0.8923334729786343
```

* 정성 평가 결과
  * AI가 변환한 코드는 ```AI converted code``` 에서 확인 가능한데, 만족스러운 수준이다.

```
 ==== original code ====
    1 | import math
    2 | def is_prime(x):
    3 |         if x == 1 or x == 2:
    4 |                 return True
    5 |         sqrt_x = int(math.sqrt(x))
    6 |         for i in range(2, sqrt_x + 1):
    7 |                 if x % i == 0:
    8 |                         return False
    9 |         return True
   10 | for i in range(1, 30):
   11 |         print(is_prime(i))

 ==== test ====
Empty new line should be between line 2 and line 3 with prob 0.995773.
Empty new line between line 3 and line 4 with prob 0.005756.
Empty new line should be between line 4 and line 5 with prob 0.983051.
Empty new line between line 5 and line 6 with prob 0.028631.
Empty new line should be between line 6 and line 7 with prob 0.987537.
Empty new line between line 7 and line 8 with prob 0.004312.
Empty new line should be between line 8 and line 9 with prob 0.98274.
Empty new line should be between line 9 and line 10 with prob 0.705336.

 ==== AI converted code ====

import math
def is_prime(x):

        if x == 1 or x == 2:
                return True

        sqrt_x = int(math.sqrt(x))
        for i in range(2, sqrt_x + 1):

                if x % i == 0:
                        return False

        return True

for i in range(1, 30):
        print(is_prime(i))
```