# model 6 (2024.03.03 20:19)

## 모델 설정 및 코드
* 모델 출력을 기존의 one-hot 방식이 아닌 embedding 방식으로, 모델의 출력 (예측 embedding 값) 과 가장 가까운 nearest embedding 찾는 방식으로 변경
* epochs = 80
* valid loss patience = 20, factor (learning rate를 해당 비율만큼 감소) = 0.8

## 모델 학습 (fit) 로그 (```train.py``` 실행)

**최초 10회 epochs**
```
Epoch 1/80
702/702 [==============================] - 38s 46ms/step - loss: 0.1247 - val_loss: 0.1149 - lr: 0.0010
Epoch 2/80
702/702 [==============================] - 44s 62ms/step - loss: 0.1140 - val_loss: 0.1123 - lr: 0.0010
Epoch 3/80
702/702 [==============================] - 44s 63ms/step - loss: 0.1073 - val_loss: 0.1110 - lr: 0.0010
Epoch 4/80
702/702 [==============================] - 41s 58ms/step - loss: 0.1024 - val_loss: 0.1116 - lr: 0.0010
Epoch 5/80
702/702 [==============================] - 46s 65ms/step - loss: 0.0982 - val_loss: 0.1140 - lr: 0.0010
Epoch 6/80
702/702 [==============================] - 42s 60ms/step - loss: 0.0946 - val_loss: 0.1144 - lr: 0.0010
Epoch 7/80
702/702 [==============================] - 45s 64ms/step - loss: 0.0912 - val_loss: 0.1152 - lr: 0.0010
Epoch 8/80
702/702 [==============================] - 45s 64ms/step - loss: 0.0888 - val_loss: 0.1162 - lr: 0.0010
Epoch 9/80
702/702 [==============================] - 46s 66ms/step - loss: 0.0865 - val_loss: 0.1164 - lr: 0.0010
Epoch 10/80
702/702 [==============================] - 45s 64ms/step - loss: 0.0843 - val_loss: 0.1174 - lr: 0.0010
```

**마지막 10회 epochs**
```
Epoch 71/80
702/702 [==============================] - 56s 80ms/step - loss: 0.0501 - val_loss: 0.1274 - lr: 5.1200e-04
Epoch 72/80
702/702 [==============================] - 55s 78ms/step - loss: 0.0500 - val_loss: 0.1276 - lr: 5.1200e-04
Epoch 73/80
702/702 [==============================] - 49s 70ms/step - loss: 0.0499 - val_loss: 0.1278 - lr: 5.1200e-04
Epoch 74/80
702/702 [==============================] - 50s 71ms/step - loss: 0.0498 - val_loss: 0.1286 - lr: 5.1200e-04
Epoch 75/80
702/702 [==============================] - 49s 70ms/step - loss: 0.0497 - val_loss: 0.1286 - lr: 5.1200e-04
Epoch 76/80
702/702 [==============================] - 49s 70ms/step - loss: 0.0495 - val_loss: 0.1278 - lr: 5.1200e-04
Epoch 77/80
702/702 [==============================] - 51s 72ms/step - loss: 0.0494 - val_loss: 0.1280 - lr: 5.1200e-04
Epoch 78/80
702/702 [==============================] - 61s 87ms/step - loss: 0.0493 - val_loss: 0.1280 - lr: 5.1200e-04
Epoch 79/80
702/702 [==============================] - 61s 87ms/step - loss: 0.0492 - val_loss: 0.1281 - lr: 5.1200e-04
Epoch 80/80
702/702 [==============================] - 60s 85ms/step - loss: 0.0492 - val_loss: 0.1289 - lr: 5.1200e-04
```

## 실험 결과 및 총평

* 기존의 one-hot 방식에 비해 **AI의 응답이 많이 어색하다.**
* **one-hot 방식으로 회귀 필요**

# model 5 (2024.03.03 13:10)

## 모델 설정 및 코드
* 데이터셋 구조, 모델 구조 등 전반적으로 변경
* epochs = 40

## 모델 학습 (fit) 로그 (```train.py``` 실행)

**최초 10회 epochs**
```
Epoch 1/40
702/702 [==============================] - 35s 42ms/step - loss: 5.8466 - val_loss: 5.6081 - lr: 0.0010
Epoch 2/40
702/702 [==============================] - 33s 46ms/step - loss: 5.1158 - val_loss: 5.4293 - lr: 0.0010
Epoch 3/40
702/702 [==============================] - 45s 64ms/step - loss: 4.6870 - val_loss: 5.4672 - lr: 0.0010
Epoch 4/40
702/702 [==============================] - 37s 53ms/step - loss: 4.3724 - val_loss: 5.5520 - lr: 0.0010
Epoch 5/40
702/702 [==============================] - 38s 54ms/step - loss: 4.1147 - val_loss: 5.7065 - lr: 0.0010
Epoch 6/40
702/702 [==============================] - 40s 57ms/step - loss: 3.8787 - val_loss: 5.9534 - lr: 0.0010
Epoch 7/40
702/702 [==============================] - 41s 58ms/step - loss: 3.6487 - val_loss: 6.1311 - lr: 0.0010
Epoch 8/40
702/702 [==============================] - 40s 57ms/step - loss: 3.4174 - val_loss: 6.5045 - lr: 0.0010
Epoch 9/40
702/702 [==============================] - 41s 58ms/step - loss: 3.1904 - val_loss: 6.8426 - lr: 0.0010
Epoch 10/40
702/702 [==============================] - 42s 59ms/step - loss: 2.9627 - val_loss: 7.0970 - lr: 0.0010
```

**마지막 10회 epochs**
```
Epoch 31/40
702/702 [==============================] - 48s 68ms/step - loss: 0.5686 - val_loss: 16.4193 - lr: 0.0010
Epoch 32/40
702/702 [==============================] - 42s 60ms/step - loss: 0.5294 - val_loss: 16.7121 - lr: 0.0010
Epoch 33/40
702/702 [==============================] - 42s 60ms/step - loss: 0.4998 - val_loss: 17.0463 - lr: 0.0010
Epoch 34/40
702/702 [==============================] - 42s 60ms/step - loss: 0.4532 - val_loss: 17.4552 - lr: 0.0010
Epoch 35/40
702/702 [==============================] - 42s 60ms/step - loss: 0.4203 - val_loss: 17.9717 - lr: 0.0010
Epoch 36/40
702/702 [==============================] - 43s 61ms/step - loss: 0.3990 - val_loss: 18.3355 - lr: 0.0010
Epoch 37/40
702/702 [==============================] - 44s 63ms/step - loss: 0.3766 - val_loss: 18.5393 - lr: 0.0010
Epoch 38/40
702/702 [==============================] - 43s 61ms/step - loss: 0.3429 - val_loss: 19.1572 - lr: 0.0010
Epoch 39/40
702/702 [==============================] - 45s 64ms/step - loss: 0.3259 - val_loss: 19.2953 - lr: 0.0010
Epoch 40/40
702/702 [==============================] - 56s 80ms/step - loss: 0.3113 - val_loss: 20.2240 - lr: 0.0010
```

## 대화 테스트 로그 (```test.py``` 실행)
현재 AI 답변은 출력된 token 을 단순히 공백으로 연결한 것이며, 실제 서비스로 배포할 때는 해당 부분을 픽스해야 합니다.

```
input text :
Hi!
AI output :
hello , how are you doing today ? busy with your day so far ?

input text :
No, I am not busy.
AI output :
talking to claim you are a scary or had it ! try this ?

input text :
I didn't try it.
AI output :
talking to chat for long . try this is diving for your hike ?

input text :
What do you mean?
AI output :
a project went to tell me . it had n't like a pretty high resolution .

input text :
What are you doing?
AI output :
i 'm go -ing to learn , but i got to get my minute schedule any month though . do you ever heard of it ?

input text :
Do you eat bread?
bread 는 vocab 에 없으므로 meat 로 대체됨
AI output :
i think so . i know a bit sad and samosas with a while . i do n't recall neural go down for the one mins driver

input text :
Hello!
AI output :
how was your party last week ?

input text :
When did you learn English?
AI output :
yes ! i made i made donuts and samosas with an air fryer have you believe in dinner food in bay area . have you seen more food . have you ever made for where we have one of ?

input text :
What are you doing now?
AI output :
at my favorite dish will we have any shovel semester ?

input text :
Which movie do you like?
AI output :
i think i know right ! i know right ! where i went to australia and finish up skiing in cats - robots in there comes as to ask .

input text :
ttyl
AI output :
how 's your day go -ing ?

input text :
I'm fine.
AI output :
how does your name ? try to keep an open mind !

input text :
Open mind?
AI output :
it 's a little a depress -ing food and something you at a try !

input text :
Okay
AI output :
corgi belly flop compilation - cute more virtual reality you like some day for his something something for christmas and ?

input text :
Merry Christmas!
merry 는 vocab 에 없으므로 giant 로 대체됨
AI output :
what is it go -ing ?

input text :
You are an artificial intelligence.
AI output :
haha . . . very honestly , i 've been to say after this sentient do to be born with . . . this stay this time , so . like is go -ing to the hotel !
```

## 총평
* AI의 답변이 비교적 자연스러운 편이지만, 질문의 의도를 파악하지 못하는 경우가 여전히 많다.
* loss를 token에 대한 categorical crossentropy에서, 자체 임베딩 모델에 근거한 실제 embedding과 예측 embedding 간의 mean squared loss로 변경하면 어떨까?
  * 이를 위해서 임베딩 검증, 자체 임베딩을 위한 첫번째 모델 학습 (기존 모델), 자체 임베딩에 근거한 embedding 예측 모델 학습 (신규 모델) 필요할듯

# model 4 (2024.03.03 05:04)

## 모델 설정 및 코드
* 데이터셋 생성 방식 변경 적용
* 이외는 **model 3** 과 모두 동일

## 모델 학습 (fit) 로그 (```train.py``` 실행)

**최초 10회 epochs**
```
Epoch 1/180
699/699 [==============================] - 57s 77ms/step - loss: 5.9118 - val_loss: 5.7094 - lr: 0.0010
Epoch 2/180
699/699 [==============================] - 62s 88ms/step - loss: 5.2957 - val_loss: 5.4968 - lr: 0.0010
Epoch 3/180
699/699 [==============================] - 59s 84ms/step - loss: 4.9160 - val_loss: 5.5363 - lr: 0.0010
Epoch 4/180
699/699 [==============================] - 61s 87ms/step - loss: 4.6447 - val_loss: 5.6069 - lr: 0.0010
Epoch 5/180
699/699 [==============================] - 55s 79ms/step - loss: 4.4237 - val_loss: 5.6599 - lr: 0.0010
Epoch 6/180
699/699 [==============================] - 55s 79ms/step - loss: 4.2384 - val_loss: 5.7974 - lr: 0.0010
Epoch 7/180
699/699 [==============================] - 55s 79ms/step - loss: 4.0784 - val_loss: 5.9082 - lr: 0.0010
Epoch 8/180
699/699 [==============================] - 55s 79ms/step - loss: 3.9075 - val_loss: 6.0698 - lr: 0.0010
Epoch 9/180
699/699 [==============================] - 56s 80ms/step - loss: 3.7626 - val_loss: 6.1821 - lr: 0.0010
Epoch 10/180
699/699 [==============================] - 55s 79ms/step - loss: 3.6111 - val_loss: 6.3420 - lr: 0.0010
```

**마지막 10회 epochs**
```
Epoch 171/180
699/699 [==============================] - 86s 123ms/step - loss: 0.5765 - val_loss: 11.1129 - lr: 6.2500e-05
Epoch 172/180
699/699 [==============================] - 92s 132ms/step - loss: 0.5743 - val_loss: 11.1167 - lr: 6.2500e-05
Epoch 173/180
699/699 [==============================] - 90s 128ms/step - loss: 0.5667 - val_loss: 11.1291 - lr: 6.2500e-05
Epoch 174/180
699/699 [==============================] - 90s 128ms/step - loss: 0.5684 - val_loss: 11.1234 - lr: 6.2500e-05
Epoch 175/180
699/699 [==============================] - 94s 135ms/step - loss: 0.5655 - val_loss: 11.1238 - lr: 6.2500e-05
Epoch 176/180
699/699 [==============================] - 95s 136ms/step - loss: 0.5705 - val_loss: 11.1218 - lr: 6.2500e-05
Epoch 177/180
699/699 [==============================] - 94s 135ms/step - loss: 0.5779 - val_loss: 11.1103 - lr: 6.2500e-05
Epoch 178/180
699/699 [==============================] - 89s 127ms/step - loss: 0.5826 - val_loss: 11.1275 - lr: 6.2500e-05
Epoch 179/180
699/699 [==============================] - 90s 129ms/step - loss: 0.5721 - val_loss: 11.1235 - lr: 6.2500e-05
Epoch 180/180
699/699 [==============================] - 92s 131ms/step - loss: 0.5631 - val_loss: 11.1194 - lr: 6.2500e-05
```

## 대화 테스트 로그 (```test.py``` 실행)
현재 AI 답변은 출력된 token 을 단순히 공백으로 연결한 것이며, 실제 서비스로 배포할 때는 해당 부분을 픽스해야 합니다.

```
input text :
What is your favorite season?
AI output :
okayish , i 'm flying by quicker than i expected . how are you up to ?

input text :
I am fine.
AI output :
what are you up to today ?

input text :
Up to?
AI output :
i 'm actual -ly a hard , but i would love hawaii .

input text :
When did you go to hawaii?
AI output :
asian food is a spinoff from a cylinder where about you ?

input text :
Cylinder?
AI output :
thank you ? do you have homemade noodle soup ?

input text :
No, I don't have it.
AI output :
i am quite decent with a laughter !

input text :
Good bye!
AI output :
i love hawaii . what types of food do you live ?

input text :
I like chicken.
AI output :
ok , very reality is real -ly a robot , i 'm close to common , yay !

input text :
What do you mean?
AI output :
i 'm a hard for a rest of the close to skills to try .

input text :
I'm OK. Thank you.
AI output :
that 's cool ! what is this saturday ?

input text :
Yesterday.
```

```
input text :
What did you do yesterday?
yesterday 는 vocab 에 없으므로 lately 로 대체됨
AI output :
i watched a few lately , but nothing is as avatar . what are you up to ?

input text :
I have nothing. What will you do today?
AI output :
i 'm real -ly enjoy the avenger movies

input text :
Avenger movies?
AI output :
yeah . maybe at a full time ! but which make you 're a hokkaido or something else ?

input text :
What is hokkaido?
AI output :
i 've only got a couple weeks haha

input text :
Ok. ttyl
AI output :
yeah that is awesome ! what do you like to ask other else ?

input text :
ttyl
AI output :
hyderabad ! i hear too much on a project . i 'm copying you well off \u0001f642 i actual -ly have a lot about different books
```

## 총평
* AI가 "Hi!" 를 출력하는 빈도가 실제로 많이 줄어들었다.
* AI가 답변을 생성하는 부분이 자연스러운 것도 맞다.
* 그러나, 질문의 의도에 맞지 않는 답변을 하는 것은 동일하기 때문에 이 부분을 개선해야 한다.
* **질문 부분을 token-wise embedding -> LSTM -> Flatten -> Embedding**, **답변 부분을 token-wise embedding -> LSTM -> Flatten -> Embedding** 각각 적용해서, 다음 token을 출력하는 것으로!!

# model 3 (2024.03.02 09:17)

## 모델 설정 및 코드
* epochs : **180**
* learning rate : ```lr_reduced = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=40, factor=0.5)```
* 이외는 **model 2** 와 모두 동일

## 모델 학습 (fit) 로그 (```train.py``` 실행)

**최초 10회 epochs**

```
Epoch 1/180
1354/1354 [==============================] - 99s 69ms/step - loss: 5.5819 - val_loss: 5.4668 - lr: 0.0010
Epoch 2/180
1354/1354 [==============================] - 98s 72ms/step - loss: 4.7870 - val_loss: 5.4226 - lr: 0.0010
Epoch 3/180
1354/1354 [==============================] - 98s 72ms/step - loss: 4.3683 - val_loss: 5.5951 - lr: 0.0010
Epoch 4/180
1354/1354 [==============================] - 101s 75ms/step - loss: 4.0769 - val_loss: 5.7212 - lr: 0.0010
Epoch 5/180
1354/1354 [==============================] - 99s 73ms/step - loss: 3.8275 - val_loss: 5.9782 - lr: 0.0010
Epoch 6/180
1354/1354 [==============================] - 97s 72ms/step - loss: 3.6204 - val_loss: 6.1820 - lr: 0.0010
Epoch 7/180
1354/1354 [==============================] - 99s 73ms/step - loss: 3.4183 - val_loss: 6.3853 - lr: 0.0010
Epoch 8/180
1354/1354 [==============================] - 100s 74ms/step - loss: 3.2522 - val_loss: 6.6129 - lr: 0.0010
Epoch 9/180
1354/1354 [==============================] - 100s 74ms/step - loss: 3.0864 - val_loss: 6.7907 - lr: 0.0010
Epoch 10/180
1354/1354 [==============================] - 100s 74ms/step - loss: 2.9450 - val_loss: 7.0368 - lr: 0.0010

...
```

**마지막 10회 epochs**

```
...

Epoch 171/180
1354/1354 [==============================] - 203s 150ms/step - loss: 0.7244 - val_loss: 10.0496 - lr: 6.2500e-05
Epoch 172/180
1354/1354 [==============================] - 298s 220ms/step - loss: 0.7293 - val_loss: 10.0447 - lr: 6.2500e-05
Epoch 173/180
1354/1354 [==============================] - 306s 226ms/step - loss: 0.7179 - val_loss: 10.0647 - lr: 6.2500e-05
Epoch 174/180
1354/1354 [==============================] - 302s 223ms/step - loss: 0.7275 - val_loss: 10.0624 - lr: 6.2500e-05
Epoch 175/180
1354/1354 [==============================] - 234s 173ms/step - loss: 0.7260 - val_loss: 10.0744 - lr: 6.2500e-05
Epoch 176/180
1354/1354 [==============================] - 333s 246ms/step - loss: 0.7081 - val_loss: 10.0861 - lr: 6.2500e-05
Epoch 177/180
1354/1354 [==============================] - 318s 235ms/step - loss: 0.7143 - val_loss: 10.0818 - lr: 6.2500e-05
Epoch 178/180
1354/1354 [==============================] - 321s 237ms/step - loss: 0.7103 - val_loss: 10.0766 - lr: 6.2500e-05
Epoch 179/180
1354/1354 [==============================] - 317s 234ms/step - loss: 0.7161 - val_loss: 10.0910 - lr: 6.2500e-05
Epoch 180/180
1354/1354 [==============================] - 310s 229ms/step - loss: 0.7233 - val_loss: 10.0896 - lr: 6.2500e-05
```

<details><summary>Click to see detailed log</summary>
<p>

```
Epoch 1/180
1354/1354 [==============================] - 99s 69ms/step - loss: 5.5819 - val_loss: 5.4668 - lr: 0.0010
Epoch 2/180
1354/1354 [==============================] - 98s 72ms/step - loss: 4.7870 - val_loss: 5.4226 - lr: 0.0010
Epoch 3/180
1354/1354 [==============================] - 98s 72ms/step - loss: 4.3683 - val_loss: 5.5951 - lr: 0.0010
Epoch 4/180
1354/1354 [==============================] - 101s 75ms/step - loss: 4.0769 - val_loss: 5.7212 - lr: 0.0010
Epoch 5/180
1354/1354 [==============================] - 99s 73ms/step - loss: 3.8275 - val_loss: 5.9782 - lr: 0.0010
Epoch 6/180
1354/1354 [==============================] - 97s 72ms/step - loss: 3.6204 - val_loss: 6.1820 - lr: 0.0010
Epoch 7/180
1354/1354 [==============================] - 99s 73ms/step - loss: 3.4183 - val_loss: 6.3853 - lr: 0.0010
Epoch 8/180
1354/1354 [==============================] - 100s 74ms/step - loss: 3.2522 - val_loss: 6.6129 - lr: 0.0010
Epoch 9/180
1354/1354 [==============================] - 100s 74ms/step - loss: 3.0864 - val_loss: 6.7907 - lr: 0.0010
Epoch 10/180
1354/1354 [==============================] - 100s 74ms/step - loss: 2.9450 - val_loss: 7.0368 - lr: 0.0010
Epoch 11/180
1354/1354 [==============================] - 99s 73ms/step - loss: 2.8241 - val_loss: 7.2300 - lr: 0.0010
Epoch 12/180
1354/1354 [==============================] - 99s 73ms/step - loss: 2.7139 - val_loss: 7.3136 - lr: 0.0010
Epoch 13/180
1354/1354 [==============================] - 102s 76ms/step - loss: 2.6182 - val_loss: 7.4241 - lr: 0.0010
Epoch 14/180
1354/1354 [==============================] - 101s 74ms/step - loss: 2.5248 - val_loss: 7.5114 - lr: 0.0010
Epoch 15/180
1354/1354 [==============================] - 102s 76ms/step - loss: 2.4460 - val_loss: 7.6126 - lr: 0.0010
Epoch 16/180
1354/1354 [==============================] - 101s 75ms/step - loss: 2.3634 - val_loss: 7.7467 - lr: 0.0010
Epoch 17/180
1354/1354 [==============================] - 101s 75ms/step - loss: 2.2983 - val_loss: 7.8785 - lr: 0.0010
Epoch 18/180
1354/1354 [==============================] - 102s 75ms/step - loss: 2.2279 - val_loss: 7.9221 - lr: 0.0010
Epoch 19/180
1354/1354 [==============================] - 102s 75ms/step - loss: 2.1658 - val_loss: 7.9685 - lr: 0.0010
Epoch 20/180
1354/1354 [==============================] - 106s 78ms/step - loss: 2.1248 - val_loss: 8.0414 - lr: 0.0010
Epoch 21/180
1354/1354 [==============================] - 102s 75ms/step - loss: 2.0677 - val_loss: 8.0629 - lr: 0.0010
Epoch 22/180
1354/1354 [==============================] - 102s 75ms/step - loss: 2.0225 - val_loss: 8.1642 - lr: 0.0010
Epoch 23/180
1354/1354 [==============================] - 106s 79ms/step - loss: 1.9807 - val_loss: 8.1751 - lr: 0.0010
Epoch 24/180
1354/1354 [==============================] - 104s 77ms/step - loss: 1.9378 - val_loss: 8.1978 - lr: 0.0010
Epoch 25/180
1354/1354 [==============================] - 107s 79ms/step - loss: 1.9007 - val_loss: 8.2794 - lr: 0.0010
Epoch 26/180
1354/1354 [==============================] - 104s 77ms/step - loss: 1.8722 - val_loss: 8.2476 - lr: 0.0010
Epoch 27/180
1354/1354 [==============================] - 105s 77ms/step - loss: 1.8347 - val_loss: 8.3742 - lr: 0.0010
Epoch 28/180
1354/1354 [==============================] - 105s 78ms/step - loss: 1.8101 - val_loss: 8.3374 - lr: 0.0010
Epoch 29/180
1354/1354 [==============================] - 106s 78ms/step - loss: 1.7765 - val_loss: 8.3506 - lr: 0.0010
Epoch 30/180
1354/1354 [==============================] - 104s 77ms/step - loss: 1.7524 - val_loss: 8.4656 - lr: 0.0010
Epoch 31/180
1354/1354 [==============================] - 106s 79ms/step - loss: 1.7256 - val_loss: 8.4006 - lr: 0.0010
Epoch 32/180
1354/1354 [==============================] - 108s 80ms/step - loss: 1.6927 - val_loss: 8.4604 - lr: 0.0010
Epoch 33/180
1354/1354 [==============================] - 106s 78ms/step - loss: 1.6723 - val_loss: 8.4281 - lr: 0.0010
Epoch 34/180
1354/1354 [==============================] - 119s 88ms/step - loss: 1.6698 - val_loss: 8.5073 - lr: 0.0010
Epoch 35/180
1354/1354 [==============================] - 110s 81ms/step - loss: 1.6315 - val_loss: 8.5560 - lr: 0.0010
Epoch 36/180
1354/1354 [==============================] - 110s 81ms/step - loss: 1.6109 - val_loss: 8.5988 - lr: 0.0010
Epoch 37/180
1354/1354 [==============================] - 110s 81ms/step - loss: 1.6043 - val_loss: 8.6524 - lr: 0.0010
Epoch 38/180
1354/1354 [==============================] - 107s 79ms/step - loss: 1.5810 - val_loss: 8.6061 - lr: 0.0010
Epoch 39/180
1354/1354 [==============================] - 107s 79ms/step - loss: 1.5593 - val_loss: 8.7041 - lr: 0.0010
Epoch 40/180
1354/1354 [==============================] - 108s 80ms/step - loss: 1.5525 - val_loss: 8.6700 - lr: 0.0010
Epoch 41/180
1354/1354 [==============================] - 111s 82ms/step - loss: 1.5210 - val_loss: 8.6910 - lr: 0.0010
Epoch 42/180
1354/1354 [==============================] - 109s 80ms/step - loss: 1.5161 - val_loss: 8.6925 - lr: 0.0010
Epoch 43/180
1354/1354 [==============================] - 111s 82ms/step - loss: 1.3987 - val_loss: 8.7974 - lr: 5.0000e-04
Epoch 44/180
1354/1354 [==============================] - 111s 82ms/step - loss: 1.3438 - val_loss: 8.8414 - lr: 5.0000e-04
Epoch 45/180
1354/1354 [==============================] - 110s 81ms/step - loss: 1.3212 - val_loss: 8.9846 - lr: 5.0000e-04
Epoch 46/180
1354/1354 [==============================] - 110s 81ms/step - loss: 1.2995 - val_loss: 8.9822 - lr: 5.0000e-04
Epoch 47/180
1354/1354 [==============================] - 112s 83ms/step - loss: 1.2841 - val_loss: 9.0363 - lr: 5.0000e-04
Epoch 48/180
1354/1354 [==============================] - 111s 82ms/step - loss: 1.2639 - val_loss: 9.0562 - lr: 5.0000e-04
Epoch 49/180
1354/1354 [==============================] - 114s 84ms/step - loss: 1.2520 - val_loss: 9.0843 - lr: 5.0000e-04
Epoch 50/180
1354/1354 [==============================] - 112s 82ms/step - loss: 1.2522 - val_loss: 9.0230 - lr: 5.0000e-04
Epoch 51/180
1354/1354 [==============================] - 112s 83ms/step - loss: 1.2407 - val_loss: 9.0479 - lr: 5.0000e-04
Epoch 52/180
1354/1354 [==============================] - 116s 85ms/step - loss: 1.2234 - val_loss: 9.1002 - lr: 5.0000e-04
Epoch 53/180
1354/1354 [==============================] - 115s 85ms/step - loss: 1.2033 - val_loss: 9.1109 - lr: 5.0000e-04
Epoch 54/180
1354/1354 [==============================] - 112s 83ms/step - loss: 1.2128 - val_loss: 9.1671 - lr: 5.0000e-04
Epoch 55/180
1354/1354 [==============================] - 118s 87ms/step - loss: 1.1946 - val_loss: 9.1332 - lr: 5.0000e-04
Epoch 56/180
1354/1354 [==============================] - 117s 87ms/step - loss: 1.1837 - val_loss: 9.2548 - lr: 5.0000e-04
Epoch 57/180
1354/1354 [==============================] - 114s 84ms/step - loss: 1.1708 - val_loss: 9.1942 - lr: 5.0000e-04
Epoch 58/180
1354/1354 [==============================] - 116s 86ms/step - loss: 1.1668 - val_loss: 9.2462 - lr: 5.0000e-04
Epoch 59/180
1354/1354 [==============================] - 117s 87ms/step - loss: 1.1696 - val_loss: 9.2315 - lr: 5.0000e-04
Epoch 60/180
1354/1354 [==============================] - 117s 86ms/step - loss: 1.1512 - val_loss: 9.2540 - lr: 5.0000e-04
Epoch 61/180
1354/1354 [==============================] - 119s 88ms/step - loss: 1.1497 - val_loss: 9.2759 - lr: 5.0000e-04
Epoch 62/180
1354/1354 [==============================] - 117s 86ms/step - loss: 1.1548 - val_loss: 9.3211 - lr: 5.0000e-04
Epoch 63/180
1354/1354 [==============================] - 120s 89ms/step - loss: 1.1370 - val_loss: 9.2521 - lr: 5.0000e-04
Epoch 64/180
1354/1354 [==============================] - 122s 90ms/step - loss: 1.1179 - val_loss: 9.2979 - lr: 5.0000e-04
Epoch 65/180
1354/1354 [==============================] - 119s 88ms/step - loss: 1.1193 - val_loss: 9.3638 - lr: 5.0000e-04
Epoch 66/180
1354/1354 [==============================] - 131s 97ms/step - loss: 1.1193 - val_loss: 9.4023 - lr: 5.0000e-04
Epoch 67/180
1354/1354 [==============================] - 138s 102ms/step - loss: 1.1087 - val_loss: 9.2959 - lr: 5.0000e-04
Epoch 68/180
1354/1354 [==============================] - 124s 91ms/step - loss: 1.1071 - val_loss: 9.3204 - lr: 5.0000e-04
Epoch 69/180
1354/1354 [==============================] - 122s 90ms/step - loss: 1.0959 - val_loss: 9.3564 - lr: 5.0000e-04
Epoch 70/180
1354/1354 [==============================] - 125s 92ms/step - loss: 1.0916 - val_loss: 9.4315 - lr: 5.0000e-04
Epoch 71/180
1354/1354 [==============================] - 126s 93ms/step - loss: 1.0890 - val_loss: 9.4531 - lr: 5.0000e-04
Epoch 72/180
1354/1354 [==============================] - 123s 91ms/step - loss: 1.0738 - val_loss: 9.4634 - lr: 5.0000e-04
Epoch 73/180
1354/1354 [==============================] - 127s 94ms/step - loss: 1.0756 - val_loss: 9.4042 - lr: 5.0000e-04
Epoch 74/180
1354/1354 [==============================] - 124s 92ms/step - loss: 1.0657 - val_loss: 9.4260 - lr: 5.0000e-04
Epoch 75/180
1354/1354 [==============================] - 126s 93ms/step - loss: 1.0547 - val_loss: 9.3960 - lr: 5.0000e-04
Epoch 76/180
1354/1354 [==============================] - 124s 91ms/step - loss: 1.0527 - val_loss: 9.4849 - lr: 5.0000e-04
Epoch 77/180
1354/1354 [==============================] - 127s 94ms/step - loss: 1.0627 - val_loss: 9.3725 - lr: 5.0000e-04
Epoch 78/180
1354/1354 [==============================] - 124s 92ms/step - loss: 1.0531 - val_loss: 9.4324 - lr: 5.0000e-04
Epoch 79/180
1354/1354 [==============================] - 128s 94ms/step - loss: 1.0408 - val_loss: 9.4735 - lr: 5.0000e-04
Epoch 80/180
1354/1354 [==============================] - 127s 94ms/step - loss: 1.0403 - val_loss: 9.4813 - lr: 5.0000e-04
Epoch 81/180
1354/1354 [==============================] - 128s 95ms/step - loss: 1.0378 - val_loss: 9.4548 - lr: 5.0000e-04
Epoch 82/180
1354/1354 [==============================] - 128s 95ms/step - loss: 1.0328 - val_loss: 9.4908 - lr: 5.0000e-04
Epoch 83/180
1354/1354 [==============================] - 131s 97ms/step - loss: 0.9758 - val_loss: 9.5410 - lr: 2.5000e-04
Epoch 84/180
1354/1354 [==============================] - 129s 95ms/step - loss: 0.9682 - val_loss: 9.5426 - lr: 2.5000e-04
Epoch 85/180
1354/1354 [==============================] - 130s 96ms/step - loss: 0.9482 - val_loss: 9.5558 - lr: 2.5000e-04
Epoch 86/180
1354/1354 [==============================] - 128s 95ms/step - loss: 0.9514 - val_loss: 9.5772 - lr: 2.5000e-04
Epoch 87/180
1354/1354 [==============================] - 130s 96ms/step - loss: 0.9332 - val_loss: 9.6057 - lr: 2.5000e-04
Epoch 88/180
1354/1354 [==============================] - 131s 97ms/step - loss: 0.9349 - val_loss: 9.6130 - lr: 2.5000e-04
Epoch 89/180
1354/1354 [==============================] - 128s 94ms/step - loss: 0.9312 - val_loss: 9.6282 - lr: 2.5000e-04
Epoch 90/180
1354/1354 [==============================] - 132s 97ms/step - loss: 0.9261 - val_loss: 9.6200 - lr: 2.5000e-04
Epoch 91/180
1354/1354 [==============================] - 130s 96ms/step - loss: 0.9199 - val_loss: 9.6522 - lr: 2.5000e-04
Epoch 92/180
1354/1354 [==============================] - 133s 98ms/step - loss: 0.9187 - val_loss: 9.6672 - lr: 2.5000e-04
Epoch 93/180
1354/1354 [==============================] - 133s 98ms/step - loss: 0.9034 - val_loss: 9.6829 - lr: 2.5000e-04
Epoch 94/180
1354/1354 [==============================] - 133s 98ms/step - loss: 0.9055 - val_loss: 9.6851 - lr: 2.5000e-04
Epoch 95/180
1354/1354 [==============================] - 138s 102ms/step - loss: 0.8921 - val_loss: 9.6870 - lr: 2.5000e-04
Epoch 96/180
1354/1354 [==============================] - 136s 100ms/step - loss: 0.9007 - val_loss: 9.7072 - lr: 2.5000e-04
Epoch 97/180
1354/1354 [==============================] - 162s 120ms/step - loss: 0.9028 - val_loss: 9.6741 - lr: 2.5000e-04
Epoch 98/180
1354/1354 [==============================] - 161s 119ms/step - loss: 0.8960 - val_loss: 9.6960 - lr: 2.5000e-04
Epoch 99/180
1354/1354 [==============================] - 157s 116ms/step - loss: 0.8777 - val_loss: 9.7227 - lr: 2.5000e-04
Epoch 100/180
1354/1354 [==============================] - 159s 118ms/step - loss: 0.8816 - val_loss: 9.7310 - lr: 2.5000e-04
Epoch 101/180
1354/1354 [==============================] - 182s 134ms/step - loss: 0.8792 - val_loss: 9.7148 - lr: 2.5000e-04
Epoch 102/180
1354/1354 [==============================] - 162s 120ms/step - loss: 0.8761 - val_loss: 9.7280 - lr: 2.5000e-04
Epoch 103/180
1354/1354 [==============================] - 167s 124ms/step - loss: 0.8754 - val_loss: 9.7756 - lr: 2.5000e-04
Epoch 104/180
1354/1354 [==============================] - 151s 111ms/step - loss: 0.8743 - val_loss: 9.7620 - lr: 2.5000e-04
Epoch 105/180
1354/1354 [==============================] - 142s 105ms/step - loss: 0.8758 - val_loss: 9.7325 - lr: 2.5000e-04
Epoch 106/180
1354/1354 [==============================] - 145s 107ms/step - loss: 0.8732 - val_loss: 9.7591 - lr: 2.5000e-04
Epoch 107/180
1354/1354 [==============================] - 144s 107ms/step - loss: 0.8630 - val_loss: 9.7522 - lr: 2.5000e-04
Epoch 108/180
1354/1354 [==============================] - 146s 108ms/step - loss: 0.8577 - val_loss: 9.7828 - lr: 2.5000e-04
Epoch 109/180
1354/1354 [==============================] - 143s 106ms/step - loss: 0.8621 - val_loss: 9.8016 - lr: 2.5000e-04
Epoch 110/180
1354/1354 [==============================] - 147s 108ms/step - loss: 0.8613 - val_loss: 9.7919 - lr: 2.5000e-04
Epoch 111/180
1354/1354 [==============================] - 148s 109ms/step - loss: 0.8577 - val_loss: 9.8073 - lr: 2.5000e-04
Epoch 112/180
1354/1354 [==============================] - 144s 107ms/step - loss: 0.8580 - val_loss: 9.7815 - lr: 2.5000e-04
Epoch 113/180
1354/1354 [==============================] - 148s 110ms/step - loss: 0.8579 - val_loss: 9.8040 - lr: 2.5000e-04
Epoch 114/180
1354/1354 [==============================] - 146s 108ms/step - loss: 0.8514 - val_loss: 9.7848 - lr: 2.5000e-04
Epoch 115/180
1354/1354 [==============================] - 148s 110ms/step - loss: 0.8519 - val_loss: 9.8102 - lr: 2.5000e-04
Epoch 116/180
1354/1354 [==============================] - 151s 111ms/step - loss: 0.8448 - val_loss: 9.7978 - lr: 2.5000e-04
Epoch 117/180
1354/1354 [==============================] - 151s 112ms/step - loss: 0.8455 - val_loss: 9.8198 - lr: 2.5000e-04
Epoch 118/180
1354/1354 [==============================] - 156s 115ms/step - loss: 0.8399 - val_loss: 9.7750 - lr: 2.5000e-04
Epoch 119/180
1354/1354 [==============================] - 153s 113ms/step - loss: 0.8408 - val_loss: 9.8067 - lr: 2.5000e-04
Epoch 120/180
1354/1354 [==============================] - 162s 120ms/step - loss: 0.8337 - val_loss: 9.8453 - lr: 2.5000e-04
Epoch 121/180
1354/1354 [==============================] - 152s 112ms/step - loss: 0.8309 - val_loss: 9.8245 - lr: 2.5000e-04
Epoch 122/180
1354/1354 [==============================] - 150s 111ms/step - loss: 0.8306 - val_loss: 9.8597 - lr: 2.5000e-04
Epoch 123/180
1354/1354 [==============================] - 153s 113ms/step - loss: 0.8082 - val_loss: 9.8690 - lr: 1.2500e-04
Epoch 124/180
1354/1354 [==============================] - 152s 112ms/step - loss: 0.8046 - val_loss: 9.8624 - lr: 1.2500e-04
Epoch 125/180
1354/1354 [==============================] - 153s 113ms/step - loss: 0.7923 - val_loss: 9.8935 - lr: 1.2500e-04
Epoch 126/180
1354/1354 [==============================] - 152s 112ms/step - loss: 0.7941 - val_loss: 9.8990 - lr: 1.2500e-04
Epoch 127/180
1354/1354 [==============================] - 154s 113ms/step - loss: 0.7930 - val_loss: 9.8707 - lr: 1.2500e-04
Epoch 128/180
1354/1354 [==============================] - 153s 113ms/step - loss: 0.7915 - val_loss: 9.8706 - lr: 1.2500e-04
Epoch 129/180
1354/1354 [==============================] - 156s 115ms/step - loss: 0.7878 - val_loss: 9.8944 - lr: 1.2500e-04
Epoch 130/180
1354/1354 [==============================] - 160s 118ms/step - loss: 0.7822 - val_loss: 9.9239 - lr: 1.2500e-04
Epoch 131/180
1354/1354 [==============================] - 157s 116ms/step - loss: 0.7961 - val_loss: 9.9276 - lr: 1.2500e-04
Epoch 132/180
1354/1354 [==============================] - 159s 117ms/step - loss: 0.7781 - val_loss: 9.9195 - lr: 1.2500e-04
Epoch 133/180
1354/1354 [==============================] - 153s 113ms/step - loss: 0.7837 - val_loss: 9.8945 - lr: 1.2500e-04
Epoch 134/180
1354/1354 [==============================] - 162s 120ms/step - loss: 0.7784 - val_loss: 9.9128 - lr: 1.2500e-04
Epoch 135/180
1354/1354 [==============================] - 159s 117ms/step - loss: 0.7750 - val_loss: 9.9555 - lr: 1.2500e-04
Epoch 136/180
1354/1354 [==============================] - 160s 118ms/step - loss: 0.7795 - val_loss: 9.9348 - lr: 1.2500e-04
Epoch 137/180
1354/1354 [==============================] - 159s 117ms/step - loss: 0.7773 - val_loss: 9.9411 - lr: 1.2500e-04
Epoch 138/180
1354/1354 [==============================] - 158s 117ms/step - loss: 0.7639 - val_loss: 9.9580 - lr: 1.2500e-04
Epoch 139/180
1354/1354 [==============================] - 160s 118ms/step - loss: 0.7693 - val_loss: 9.9662 - lr: 1.2500e-04
Epoch 140/180
1354/1354 [==============================] - 161s 119ms/step - loss: 0.7741 - val_loss: 9.9506 - lr: 1.2500e-04
Epoch 141/180
1354/1354 [==============================] - 162s 120ms/step - loss: 0.7739 - val_loss: 9.9500 - lr: 1.2500e-04
Epoch 142/180
1354/1354 [==============================] - 163s 120ms/step - loss: 0.7784 - val_loss: 9.9575 - lr: 1.2500e-04
Epoch 143/180
1354/1354 [==============================] - 163s 121ms/step - loss: 0.7759 - val_loss: 9.9560 - lr: 1.2500e-04
Epoch 144/180
1354/1354 [==============================] - 167s 124ms/step - loss: 0.7693 - val_loss: 9.9722 - lr: 1.2500e-04
Epoch 145/180
1354/1354 [==============================] - 163s 120ms/step - loss: 0.7720 - val_loss: 9.9831 - lr: 1.2500e-04
Epoch 146/180
1354/1354 [==============================] - 164s 121ms/step - loss: 0.7677 - val_loss: 9.9770 - lr: 1.2500e-04
Epoch 147/180
1354/1354 [==============================] - 166s 123ms/step - loss: 0.7601 - val_loss: 9.9912 - lr: 1.2500e-04
Epoch 148/180
1354/1354 [==============================] - 165s 122ms/step - loss: 0.7689 - val_loss: 9.9979 - lr: 1.2500e-04
Epoch 149/180
1354/1354 [==============================] - 173s 128ms/step - loss: 0.7654 - val_loss: 9.9848 - lr: 1.2500e-04
Epoch 150/180
1354/1354 [==============================] - 168s 124ms/step - loss: 0.7584 - val_loss: 9.9915 - lr: 1.2500e-04
Epoch 151/180
1354/1354 [==============================] - 170s 126ms/step - loss: 0.7583 - val_loss: 9.9863 - lr: 1.2500e-04
Epoch 152/180
1354/1354 [==============================] - 171s 126ms/step - loss: 0.7552 - val_loss: 10.0097 - lr: 1.2500e-04
Epoch 153/180
1354/1354 [==============================] - 173s 128ms/step - loss: 0.7475 - val_loss: 10.0022 - lr: 1.2500e-04
Epoch 154/180
1354/1354 [==============================] - 187s 138ms/step - loss: 0.7487 - val_loss: 9.9948 - lr: 1.2500e-04
Epoch 155/180
1354/1354 [==============================] - 183s 135ms/step - loss: 0.7592 - val_loss: 10.0078 - lr: 1.2500e-04
Epoch 156/180
1354/1354 [==============================] - 177s 131ms/step - loss: 0.7433 - val_loss: 10.0248 - lr: 1.2500e-04
Epoch 157/180
1354/1354 [==============================] - 224s 166ms/step - loss: 0.7414 - val_loss: 10.0169 - lr: 1.2500e-04
Epoch 158/180
1354/1354 [==============================] - 187s 138ms/step - loss: 0.7493 - val_loss: 10.0128 - lr: 1.2500e-04
Epoch 159/180
1354/1354 [==============================] - 212s 157ms/step - loss: 0.7542 - val_loss: 10.0067 - lr: 1.2500e-04
Epoch 160/180
1354/1354 [==============================] - 190s 140ms/step - loss: 0.7533 - val_loss: 10.0399 - lr: 1.2500e-04
Epoch 161/180
1354/1354 [==============================] - 205s 152ms/step - loss: 0.7427 - val_loss: 10.0348 - lr: 1.2500e-04
Epoch 162/180
1354/1354 [==============================] - 201s 148ms/step - loss: 0.7398 - val_loss: 10.0373 - lr: 1.2500e-04
Epoch 163/180
1354/1354 [==============================] - 219s 162ms/step - loss: 0.7423 - val_loss: 10.0433 - lr: 6.2500e-05
Epoch 164/180
1354/1354 [==============================] - 248s 183ms/step - loss: 0.7324 - val_loss: 10.0448 - lr: 6.2500e-05
Epoch 165/180
1354/1354 [==============================] - 244s 181ms/step - loss: 0.7233 - val_loss: 10.0558 - lr: 6.2500e-05
Epoch 166/180
1354/1354 [==============================] - 215s 159ms/step - loss: 0.7232 - val_loss: 10.0520 - lr: 6.2500e-05
Epoch 167/180
1354/1354 [==============================] - 199s 147ms/step - loss: 0.7271 - val_loss: 10.0456 - lr: 6.2500e-05
Epoch 168/180
1354/1354 [==============================] - 210s 155ms/step - loss: 0.7263 - val_loss: 10.0534 - lr: 6.2500e-05
Epoch 169/180
1354/1354 [==============================] - 290s 214ms/step - loss: 0.7275 - val_loss: 10.0625 - lr: 6.2500e-05
Epoch 170/180
1354/1354 [==============================] - 249s 184ms/step - loss: 0.7227 - val_loss: 10.0647 - lr: 6.2500e-05
Epoch 171/180
1354/1354 [==============================] - 203s 150ms/step - loss: 0.7244 - val_loss: 10.0496 - lr: 6.2500e-05
Epoch 172/180
1354/1354 [==============================] - 298s 220ms/step - loss: 0.7293 - val_loss: 10.0447 - lr: 6.2500e-05
Epoch 173/180
1354/1354 [==============================] - 306s 226ms/step - loss: 0.7179 - val_loss: 10.0647 - lr: 6.2500e-05
Epoch 174/180
1354/1354 [==============================] - 302s 223ms/step - loss: 0.7275 - val_loss: 10.0624 - lr: 6.2500e-05
Epoch 175/180
1354/1354 [==============================] - 234s 173ms/step - loss: 0.7260 - val_loss: 10.0744 - lr: 6.2500e-05
Epoch 176/180
1354/1354 [==============================] - 333s 246ms/step - loss: 0.7081 - val_loss: 10.0861 - lr: 6.2500e-05
Epoch 177/180
1354/1354 [==============================] - 318s 235ms/step - loss: 0.7143 - val_loss: 10.0818 - lr: 6.2500e-05
Epoch 178/180
1354/1354 [==============================] - 321s 237ms/step - loss: 0.7103 - val_loss: 10.0766 - lr: 6.2500e-05
Epoch 179/180
1354/1354 [==============================] - 317s 234ms/step - loss: 0.7161 - val_loss: 10.0910 - lr: 6.2500e-05
Epoch 180/180
1354/1354 [==============================] - 310s 229ms/step - loss: 0.7233 - val_loss: 10.0896 - lr: 6.2500e-05
Model: "mini_chat_gpt_model"
```

</p>
</details>

## 다음 토큰 순위 테스트 로그 (```train.py``` 실행)
```
tokens: ['what', 'was', 'the', 'most', 'number', 'of', 'people', 'you', 'have', 'ever', 'met', 'during', 'a', 'working', 'day', '', '?', '<person-change>']
ID of tokens: [2708, 2681, 2463, 1565, 1637, 1651, 1772, 2782, 1106, 814, 1521, 752, 71, 2750, 640, 0, 57, 53]
ID of tokens: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 2708, 2681, 2463, 1565, 1637, 1651, 1772, 2782, 1106, 814, 1521, 752, 71, 2750, 640, 0, 57, 53]
mini chatgpt model output: [2.4060717e-05 7.7519542e-07 1.1391905e-10 ... 4.6002338e-08 3.3020171e-12
 1.0592481e-08]
first 10 of token arr: ['', '!', '!--', '!unlike', "'billions'", "'court'", "'d", "'m", "'re", "'s"]
rank 0 : ['i', 0.5377755165100098]
rank 1 : ['well', 0.13132335245609283]
rank 2 : ['no', 0.10410797595977783]
rank 3 : ['nope', 0.044317640364170074]
rank 4 : ['we', 0.03635754808783531]
rank 5 : ['yes', 0.028941437602043152]
rank 6 : ['hard', 0.02113773301243782]
rank 7 : ['you', 0.01069169957190752]
rank 8 : ['they', 0.009079891256988049]
rank 9 : ['it', 0.007970258593559265]
rank 10 : ['either', 0.004981290549039841]
rank 11 : ['not', 0.004964393097907305]
rank 12 : ['a', 0.004805349744856358]
rank 13 : ['the', 0.00447991956025362]
rank 14 : ['that', 0.004211819730699062]
rank 15 : ['yeah', 0.004112060181796551]
rank 16 : ['lol', 0.003953986801207066]
rank 17 : ['at', 0.0037380082067102194]
rank 18 : ['admit', 0.003561679506674409]
rank 19 : ['my', 0.0035069489385932684]

tokens: ['i', 'know', 'him', 'very', 'well', '', '.', '<person-change>', 'is', 'him', 'your', 'friend', '', '?', 'if', 'so', '', ',', 'it', '<person-change>']
ID of tokens: [1204, 1343, 1144, 2643, 2704, 0, 21, 53, 1280, 1144, 2784, 975, 0, 57, 1218, 2220, 0, 16, 1283, 53]
ID of tokens: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 1204, 1343, 1144, 2643, 2704, 0, 21, 53, 1280, 1144, 2784, 975, 0, 57, 1218, 2220, 0, 16, 1283, 53]
mini chatgpt model output: [1.1120834e-05 2.4349336e-04 3.5760427e-06 ... 2.0440252e-10 1.7486986e-09
 2.7364541e-07]
first 10 of token arr: ['', '!', '!--', '!unlike', "'billions'", "'court'", "'d", "'m", "'re", "'s"]
rank 0 : ['i', 0.23438815772533417]
rank 1 : ['good', 0.11543290317058563]
rank 2 : ['seems', 0.08586568385362625]
rank 3 : ['hi', 0.08000662177801132]
rank 4 : ['will', 0.036834556609392166]
rank 5 : ['cuban', 0.03600301593542099]
rank 6 : ['sounds', 0.034278061240911484]
rank 7 : ['that', 0.03251875564455986]
rank 8 : ['can', 0.02528771571815014]
rank 9 : ['no', 0.022277355194091797]
rank 10 : ["'s", 0.02090260572731495]
rank 11 : ['does', 0.019992252811789513]
rank 12 : ['not', 0.016337506473064423]
rank 13 : ['one', 0.012500611133873463]
rank 14 : ['should', 0.011235641315579414]
rank 15 : ['do', 0.00992217194288969]
rank 16 : ['is', 0.009800134226679802]
rank 17 : ['have', 0.00933048129081726]
rank 18 : ['what', 0.007966122590005398]
rank 19 : ['a', 0.00788514968007803]

tokens: ['how', 'can', 'i', 'do', 'for', 'you', '', '?', '<person-change>', 'can', 'you', 'buy', 'me', 'a', 'book', '', '?', '<person-change>']
ID of tokens: [1184, 381, 1204, 713, 951, 2782, 0, 57, 53, 381, 2782, 365, 1485, 71, 311, 0, 57, 53]
ID of tokens: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 1184, 381, 1204, 713, 951, 2782, 0, 57, 53, 381, 2782, 365, 1485, 71, 311, 0, 57, 53]
mini chatgpt model output: [1.4124701e-05 1.0584735e-05 1.2024249e-06 ... 5.8132926e-11 6.1469424e-10
 1.1433486e-06]
first 10 of token arr: ['', '!', '!--', '!unlike', "'billions'", "'court'", "'d", "'m", "'re", "'s"]
rank 0 : ['hi', 0.33098113536834717]
rank 1 : ['thats', 0.3034041225910187]
rank 2 : ['lol', 0.09800108522176743]
rank 3 : ['true', 0.08416243642568588]
rank 4 : ['darn', 0.02079092711210251]
rank 5 : ['sure', 0.01633305288851261]
rank 6 : ['wait', 0.015407280996441841]
rank 7 : ['oh', 0.015274367295205593]
rank 8 : ['yup', 0.014483473263680935]
rank 9 : ['thanks', 0.012106266804039478]
rank 10 : ['that', 0.010151137597858906]
rank 11 : ['haha', 0.007197904866188765]
rank 12 : ['wow', 0.0070889173075556755]
rank 13 : ['i', 0.006931590847671032]
rank 14 : ['cool', 0.005991070996969938]
rank 15 : ['a', 0.005144206341356039]
rank 16 : ['definitely', 0.003913815598934889]
rank 17 : ['\\u0001f602', 0.003393589984625578]
rank 18 : ['no', 0.003216576762497425]
rank 19 : ['yes', 0.002898682840168476]
```

## 대화 테스트 로그 (```test.py``` 실행)
현재 AI 답변은 출력된 token 을 단순히 공백으로 연결한 것이며, 실제 서비스로 배포할 때는 해당 부분을 픽스해야 합니다.
```
input text :
What is your favorite season?
AI output :
i real -ly enjoy barley tea what about you ?

input text :
Is barley tea delicious?
AI output :
hi !

input text :
Hello!
AI output :
hello , how are you doing today ?

input text :
I am fine. You too?
AI output :
a little bit tired so i got super super way on my morning . what was your favorite movie ?

input text :
I like Iron man movies.
error: 'iron'
all outputs: i real -ly enjoy barley tea what about you ? <person-change> hi ! <person-change> hello , how are you doing today ? <person-change> a little bit tired so i got super super way on my morning . what was your favorite movie ? <person-change>
```

```
input text :
Nice to meet you!
AI output :
ttyl !

input text :
What are you doing?
AI output :
i signed up a few years ago to help people process the house and asks not big italian . ever heard of it ?

input text :
I didn't heard. What is it?
AI output :
it 's an iconic place ! we do a lot of folks practice for the holidays .

input text :
For example?
AI output :
i 've gotten real -ly into yoga lately . i went to a class today and i am doing my back as i went to the local cafe and had a double espresso . it was perfect .

input text :
Was double espresso you had delicious?
AI output :
hi !

input text :
Was double espresso you had delicious?
AI output :
hi !

input text :
Don't you understand?
AI output :
i do n't have to learn guitar .

input text :
Why do you learn guitar?
AI output :
yeah i real -ly went to an onsen , the fish market , disney land and giant robot fighting show haha

input text :
Was the show interesting?
AI output :
right near a high school !

input text :
Yes I see.
AI output :
hi !

input text :
Good bye!
AI output :
hi !
```

## 총평
지난 train loss 최종 2.8 이었던 모델, train loss 최종 1.6 이었던 모델에 비해, 이번 **train loss 최종 0.72** 인 모델은 **챗봇의 응답이 더 자연스럽고, 인사 등 간단한 질문에 대해서는 그 의도도 조금씩 파악하는 듯하다. 그러나 아직 부족하다.**

TODO
* 유사한 과제 (next token prediction for Chatbot) 에서 사람들이 많이 활용한 RMSProp 으로 마찬가지로 180 epochs로 실험 (3월 3일 새벽)
  * RMSProp Loss 관련 참고 사항 : [딥러닝 기초 Optimizer 정리 문서](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Deep%20Learning%20Basics/%EB%94%A5%EB%9F%AC%EB%8B%9D_%EA%B8%B0%EC%B4%88_Optimizer.md)
  * train loss가 발산하는지, 발산하는 상황에서도 대화가 자연스러운지 체크
* learning rate 감소 주기를 ```patience=40``` 에서 ```patience=16``` 으로 조정하여 재실험
* 문맥 token 길이를 36에서 60 정도로 늘려서 실험 (null 토큰을 추가하는 부분은 전체 토큰 개수의 1/6 정도만 null이 아니게 하면 되는 것은 유지하지만, null 토큰을 포함한 학습 데이터를 기존보다 줄이자!)
  * RMSProp으로 train loss가 수렴하거나, 대화가 자연스러울 시 RMSProp을 이용, 그렇지 않으면 Adam 또는 AdamW를 이용

# model 2 (2024.03.02 01:31)

## 모델 설정 및 코드

* epochs : **40**

```
INPUT_TOKEN_CNT = 36 # 학습 데이터 row 당 입력 토큰 개수
TKN_EMBEDDING_DIM = 128 # token embedding dimension
VOCAB_SIZE = None


# mini chatgpt model (=NLP model)
# ref: https://www.kaggle.com/code/carlosaguayo/predicting-the-next-word-using-lstm/notebook
class MiniChatGPTModel(tf.keras.Model):
    
    def __init__(self, dropout_rate=0.45):
        super().__init__()
        global VOCAB_SIZE, INPUT_TOKEN_CNT

        L2 = tf.keras.regularizers.l2(0.001)
        self.dropout = Dropout(rate=dropout_rate)

        # token embedding
        self.tkn_embedding = Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=TKN_EMBEDDING_DIM,
            input_length=INPUT_TOKEN_CNT
        )
        
        self.BIDRC_LSTM_0 = Bidirectional(LSTM(128))
        self.dense = Dense(192, activation=LeakyReLU(alpha=0.1))
        self.final = Dense(VOCAB_SIZE, activation='softmax')
```

## 모델 구조 로그 (```train.py``` 실행)

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dropout (Dropout)           multiple                  0

 embedding (Embedding)       multiple                  358528

 bidirectional (Bidirectiona  multiple                 263168
 l)

 dense (Dense)               multiple                  49344

 dense_1 (Dense)             multiple                  540593

=================================================================
Total params: 1,211,633
Trainable params: 1,211,633
Non-trainable params: 0
_________________________________________________________________
```

## 모델 학습 (fit) 로그 (```train.py``` 실행)

```
Epoch 1/40
1354/1354 [==============================] - 110s 79ms/step - loss: 5.5964 - val_loss: 5.4930 - lr: 0.0010
Epoch 2/40
1354/1354 [==============================] - 120s 89ms/step - loss: 4.8276 - val_loss: 5.4540 - lr: 0.0010
Epoch 3/40
1354/1354 [==============================] - 123s 91ms/step - loss: 4.4052 - val_loss: 5.5454 - lr: 0.0010
Epoch 4/40
1354/1354 [==============================] - 125s 93ms/step - loss: 4.1011 - val_loss: 5.8000 - lr: 0.0010
Epoch 5/40
1354/1354 [==============================] - 125s 92ms/step - loss: 3.8695 - val_loss: 5.9613 - lr: 0.0010
Epoch 6/40
1354/1354 [==============================] - 124s 91ms/step - loss: 3.6643 - val_loss: 6.2413 - lr: 0.0010
Epoch 7/40
1354/1354 [==============================] - 125s 92ms/step - loss: 3.4687 - val_loss: 6.4388 - lr: 0.0010
Epoch 8/40
1354/1354 [==============================] - 124s 92ms/step - loss: 3.2984 - val_loss: 6.6436 - lr: 0.0010
Epoch 9/40
1354/1354 [==============================] - 124s 92ms/step - loss: 3.1453 - val_loss: 6.8415 - lr: 0.0010
Epoch 10/40
1354/1354 [==============================] - 125s 93ms/step - loss: 3.0047 - val_loss: 7.0123 - lr: 0.0010
Epoch 11/40
1354/1354 [==============================] - 132s 97ms/step - loss: 2.8779 - val_loss: 7.1268 - lr: 0.0010
Epoch 12/40
1354/1354 [==============================] - 126s 93ms/step - loss: 2.7606 - val_loss: 7.2398 - lr: 0.0010
Epoch 13/40
1354/1354 [==============================] - 144s 106ms/step - loss: 2.6625 - val_loss: 7.3784 - lr: 0.0010
Epoch 14/40
1354/1354 [==============================] - 130s 96ms/step - loss: 2.5694 - val_loss: 7.5040 - lr: 0.0010
Epoch 15/40
1354/1354 [==============================] - 127s 94ms/step - loss: 2.4791 - val_loss: 7.6720 - lr: 0.0010
Epoch 16/40
1354/1354 [==============================] - 127s 94ms/step - loss: 2.4221 - val_loss: 7.6838 - lr: 0.0010
Epoch 17/40
1354/1354 [==============================] - 129s 95ms/step - loss: 2.3345 - val_loss: 7.7547 - lr: 0.0010
Epoch 18/40
1354/1354 [==============================] - 127s 94ms/step - loss: 2.2764 - val_loss: 7.7850 - lr: 0.0010
Epoch 19/40
1354/1354 [==============================] - 128s 95ms/step - loss: 2.2199 - val_loss: 7.8314 - lr: 0.0010
Epoch 20/40
1354/1354 [==============================] - 128s 95ms/step - loss: 2.1674 - val_loss: 7.8727 - lr: 0.0010
Epoch 21/40
1354/1354 [==============================] - 127s 94ms/step - loss: 2.1174 - val_loss: 7.9630 - lr: 0.0010
Epoch 22/40
1354/1354 [==============================] - 129s 95ms/step - loss: 2.0737 - val_loss: 8.0221 - lr: 0.0010
Epoch 23/40
1354/1354 [==============================] - 129s 95ms/step - loss: 2.0418 - val_loss: 8.0093 - lr: 0.0010
Epoch 24/40
1354/1354 [==============================] - 130s 96ms/step - loss: 1.9986 - val_loss: 8.0408 - lr: 0.0010
Epoch 25/40
1354/1354 [==============================] - 128s 95ms/step - loss: 1.9658 - val_loss: 8.1044 - lr: 0.0010
Epoch 26/40
1354/1354 [==============================] - 128s 94ms/step - loss: 1.9352 - val_loss: 8.1327 - lr: 0.0010
Epoch 27/40
1354/1354 [==============================] - 130s 96ms/step - loss: 1.8954 - val_loss: 8.2496 - lr: 0.0010
Epoch 28/40
1354/1354 [==============================] - 131s 97ms/step - loss: 1.8743 - val_loss: 8.2101 - lr: 0.0010
Epoch 29/40
1354/1354 [==============================] - 130s 96ms/step - loss: 1.8351 - val_loss: 8.2960 - lr: 0.0010
Epoch 30/40
1354/1354 [==============================] - 129s 95ms/step - loss: 1.8080 - val_loss: 8.2845 - lr: 0.0010
Epoch 31/40
1354/1354 [==============================] - 129s 95ms/step - loss: 1.7904 - val_loss: 8.3957 - lr: 0.0010
Epoch 32/40
1354/1354 [==============================] - 130s 96ms/step - loss: 1.7684 - val_loss: 8.3679 - lr: 0.0010
Epoch 33/40
1354/1354 [==============================] - 129s 95ms/step - loss: 1.7345 - val_loss: 8.4442 - lr: 0.0010
Epoch 34/40
1354/1354 [==============================] - 130s 96ms/step - loss: 1.7244 - val_loss: 8.4211 - lr: 0.0010
Epoch 35/40
1354/1354 [==============================] - 130s 96ms/step - loss: 1.7014 - val_loss: 8.4375 - lr: 0.0010
Epoch 36/40
1354/1354 [==============================] - 133s 99ms/step - loss: 1.6723 - val_loss: 8.4351 - lr: 0.0010
Epoch 37/40
1354/1354 [==============================] - 117s 86ms/step - loss: 1.6644 - val_loss: 8.4619 - lr: 0.0010
Epoch 38/40
1354/1354 [==============================] - 117s 86ms/step - loss: 1.6419 - val_loss: 8.3760 - lr: 0.0010
Epoch 39/40
1354/1354 [==============================] - 118s 87ms/step - loss: 1.6255 - val_loss: 8.5110 - lr: 0.0010
Epoch 40/40
1354/1354 [==============================] - 120s 89ms/step - loss: 1.6123 - val_loss: 8.4779 - lr: 0.0010
```

## 다음 토큰 순위 테스트 로그 (```train.py``` 실행)
```
tokens: ['what', 'was', 'the', 'most', 'number', 'of', 'people', 'you', 'have', 'ever', 'met', 'during', 'a', 'working', 'day', '', '?', '<person-change>']
ID of tokens: [2708, 2681, 2463, 1565, 1637, 1651, 1772, 2782, 1106, 814, 1521, 752, 71, 2750, 640, 0, 57, 53]
ID of tokens: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 2708, 2681, 2463, 1565, 1637, 1651, 1772, 2782, 1106, 814, 1521, 752, 71, 2750, 640, 0, 57, 53]
mini chatgpt model output: [2.3578769e-03 1.2832024e-04 8.1298509e-07 ... 9.2945567e-07 1.1358058e-08
 6.0701797e-07]
first 10 of token arr: ['', '!', '!--', '!unlike', "'billions'", "'court'", "'d", "'m", "'re", "'s"]
rank 0 : ['yeah', 0.1982189416885376]
rank 1 : ['i', 0.13240747153759003]
rank 2 : ['no', 0.1250891387462616]
rank 3 : ['yes', 0.07773283869028091]
rank 4 : ['hmmmm', 0.04164377972483635]
rank 5 : ['ah', 0.03754977136850357]
rank 6 : ['oh', 0.02619194984436035]
rank 7 : ['true', 0.01907411217689514]
rank 8 : ['admit', 0.017851844429969788]
rank 9 : ['noodle', 0.017426161095499992]
rank 10 : ['we', 0.01730366051197052]
rank 11 : ['nope', 0.017002221196889877]
rank 12 : ['sure', 0.015324749052524567]
rank 13 : ['depends', 0.013863597065210342]
rank 14 : ['hi', 0.013665417209267616]
rank 15 : ['lol', 0.013634657487273216]
rank 16 : ['well', 0.013099875301122665]
rank 17 : ['not', 0.01284028124064207]
rank 18 : ['haha', 0.011791757307946682]
rank 19 : ['likewise', 0.010808155871927738]

tokens: ['i', 'know', 'him', 'very', 'well', '', '.', '<person-change>', 'is', 'him', 'your', 'friend', '', '?', 'if', 'so', '', ',', 'it', '<person-change>']
ID of tokens: [1204, 1343, 1144, 2643, 2704, 0, 21, 53, 1280, 1144, 2784, 975, 0, 57, 1218, 2220, 0, 16, 1283, 53]
ID of tokens: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 1204, 1343, 1144, 2643, 2704, 0, 21, 53, 1280, 1144, 2784, 975, 0, 57, 1218, 2220, 0, 16, 1283, 53]
mini chatgpt model output: [3.4006269e-04 6.5233246e-05 4.5032471e-07 ... 6.1471164e-06 1.5262127e-09
 2.6960741e-08]
first 10 of token arr: ['', '!', '!--', '!unlike', "'billions'", "'court'", "'d", "'m", "'re", "'s"]
rank 0 : ['hi', 0.14377261698246002]
rank 1 : ['oh', 0.08038187772035599]
rank 2 : ['im', 0.06189049780368805]
rank 3 : ['ouch', 0.05749477446079254]
rank 4 : ['i', 0.05586232990026474]
rank 5 : ['yeah', 0.052465785294771194]
rank 6 : ['seems', 0.03977170214056969]
rank 7 : ['cool', 0.03355791047215462]
rank 8 : ['awesome', 0.031132513657212257]
rank 9 : ['nope', 0.02840917930006981]
rank 10 : ['yes', 0.027056748047471046]
rank 11 : ['does', 0.02178298868238926]
rank 12 : ['haha', 0.020056169480085373]
rank 13 : ['either', 0.017450332641601562]
rank 14 : ['arghhh', 0.015636183321475983]
rank 15 : ['all', 0.014835222624242306]
rank 16 : ['thats', 0.0146075040102005]
rank 17 : ['true', 0.012603809125721455]
rank 18 : ['what', 0.012580029666423798]
rank 19 : ['yea', 0.01046320702880621]

tokens: ['how', 'can', 'i', 'do', 'for', 'you', '', '?', '<person-change>', 'can', 'you', 'buy', 'me', 'a', 'book', '', '?', '<person-change>']
ID of tokens: [1184, 381, 1204, 713, 951, 2782, 0, 57, 53, 381, 2782, 365, 1485, 71, 311, 0, 57, 53]
ID of tokens: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 1184, 381, 1204, 713, 951, 2782, 0, 57, 53, 381, 2782, 365, 1485, 71, 311, 0, 57, 53]
mini chatgpt model output: [3.0183161e-04 9.8760902e-05 2.6442491e-08 ... 9.5166841e-05 2.0999481e-10
 3.8253001e-09]
first 10 of token arr: ['', '!', '!--', '!unlike', "'billions'", "'court'", "'d", "'m", "'re", "'s"]
rank 0 : ['i', 0.21374785900115967]
rank 1 : ['nope', 0.18364255130290985]
rank 2 : ['yes', 0.11089217662811279]
rank 3 : ['no', 0.1076144352555275]
rank 4 : ['yeah', 0.10562951117753983]
rank 5 : ['hmm', 0.03441639244556427]
rank 6 : ['hi', 0.031036093831062317]
rank 7 : ['oh', 0.018554385751485825]
rank 8 : ['not', 0.011347801424562931]
rank 9 : ['well', 0.010891283862292767]
rank 10 : ['haha', 0.009984000585973263]
rank 11 : ['it', 0.008660438470542431]
rank 12 : ['anything', 0.008111309260129929]
rank 13 : ['a', 0.007332478184252977]
rank 14 : ['we', 0.006951083429157734]
rank 15 : ['that', 0.00598540622740984]
rank 16 : ['upstate', 0.005838797893375158]
rank 17 : ['barely', 0.005588383413851261]
rank 18 : ['pictionary', 0.004646633751690388]
rank 19 : ['im', 0.004388262517750263]
```

## 대화 테스트 로그 (```test.py``` 실행)
현재 AI 답변은 출력된 token 을 단순히 공백으로 연결한 것이며, 실제 서비스로 배포할 때는 해당 부분을 픽스해야 합니다.

```
input text :
What is your favorite season?
AI output :
i trained in contemporary and jazz .

input text :
Contemporary and Jazz?
AI output :
haha , apart from during the friendship i expected . how has your favorite cities thing for society ?

input text :
My favorite city?
AI output :
hmm , popcorn for the sequels .

input text :
How are you?
AI output :
i 've seen avatar too much . what do you do anything ?

input text :
I am listening to musics.
error: 'musics'
all outputs: i trained in contemporary and jazz . <person-change> haha , apart from during the friendship i expected . how has your favorite cities thing for society ? <person-change> hmm , popcorn for the sequels . <person-change> i 've seen avatar too much . what do you do anything ? <person-change>
```

```
input text :
Hello!
AI output :
hi !

input text :
Nice to meet you!
AI output :
hi !

input text :
Do you know this music?
AI output :
i visited maui . it 's a lot of helpless , sad papers fighting fish . actual -ly france , vita nostra and dark matter ) but i got promoted next cycle i formed go somewhere for the year .

input text :
Where will you go for the year?
AI output :
i am exhausted by a farm . helping three flavors ?

input text :
Are you busy now?
AI output :
i loved the x-men as a kid , but i useful as well not a bit shameful to both and and i do n't understand .

input text :
What do you mean?
AI output :
i used logic pro x , pretty handy stuff as you 're a rockstar !

input text :
Am I a rockstar then?
AI output :
nope , i 'm a lawyer . i feel like god

input text :
Wow that's great!
AI output :
hi !

input text :
It's time to go to bed.
error: 'bed'
all outputs: hi ! <person-change> hi ! <person-change> i visited maui . it 's a lot of helpless , sad papers fighting fish . actual -ly france , vita nostra and dark matter ) but i got promoted next cycle i formed go somewhere for the year . <person-change> i am exhausted by a farm . helping three flavors ? <person-change> i loved the x-men as a kid , but i useful as well not a bit shameful to both and and i do n't understand . <person-change> i used logic pro x , pretty handy stuff as you 're a rockstar ! <person-change> nope , i 'm a lawyer . i feel like god <person-change> hi ! <person-change>
```

```
input text :
Good bye!
AI output :
hey sup

input text :
Hey! Don't you understand?
AI output :
i 'm jealous . sorry for the late response -- i traveled around a delicious over the weekends .

input text :
Delicious? Where did you travel?
AI output :
i love noodle soups who have n't do plenty of turkey

input text :
Do you want to eat them?
AI output :
oh i grew up speaking it . it sucks to tell me a balance . why we do the count in thrones
```

## 총평
train loss 1.6 부근에서 종료한 모델로, 기존 train loss 2.8 부근에서 종료한 모델보다 **전반적으로 답변이 자연스럽다. 단, 질문의 의도에 적절한 답변을 하는 능력은 여전히 다소 부족하다.**

**TODO**
* epochs 40 -> 160 으로 테스트 시도
* RMSProp Loss 적용 테스트 시도 (valid loss뿐만 아니라 train loss 마저 발산하면서 높아질 것으로 예상되는데, 이런 상황에서도 실제 AI Chatbot 과의 채팅이 자연스럽게 이루어질 수 있을지도?)

# model 1 (2024.03.01 23:07)

## 모델 설정 및 코드

* epochs : **40**

```
INPUT_TOKEN_CNT = 36 # 학습 데이터 row 당 입력 토큰 개수
TKN_EMBEDDING_DIM = 96 # token embedding dimension
VOCAB_SIZE = None


# mini chatgpt model (=NLP model)
# ref: https://www.kaggle.com/code/carlosaguayo/predicting-the-next-word-using-lstm/notebook
class MiniChatGPTModel(tf.keras.Model):
    
    def __init__(self, dropout_rate=0.45):
        super().__init__()
        global VOCAB_SIZE, INPUT_TOKEN_CNT

        L2 = tf.keras.regularizers.l2(0.001)
        self.dropout = Dropout(rate=dropout_rate)

        # token embedding
        self.tkn_embedding = Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=TKN_EMBEDDING_DIM,
            input_length=INPUT_TOKEN_CNT
        )
        
        self.BIDRC_LSTM_0 = Bidirectional(LSTM(96, return_sequences=True))
        self.BIDRC_LSTM_1 = Bidirectional(LSTM(96))
        self.dense = Dense(128, activation=LeakyReLU(alpha=0.1))
        self.final = Dense(VOCAB_SIZE, activation='softmax')
```

## 모델 구조 로그 (```train.py``` 실행)

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dropout (Dropout)           multiple                  0

 embedding (Embedding)       multiple                  268896

 bidirectional (Bidirectiona  multiple                 148224
 l)

 bidirectional_1 (Bidirectio  multiple                 221952
 nal)

 dense (Dense)               multiple                  24704

 dense_1 (Dense)             multiple                  361329

=================================================================
Total params: 1,025,105
Trainable params: 1,025,105
Non-trainable params: 0
_________________________________________________________________
```

## 모델 학습 (fit) 로그 (```train.py``` 실행)

```
Epoch 1/40
1354/1354 [==============================] - 165s 118ms/step - loss: 5.6866 - val_loss: 5.5543 - lr: 0.0010
Epoch 2/40
1354/1354 [==============================] - 169s 124ms/step - loss: 5.0515 - val_loss: 5.5871 - lr: 0.0010
Epoch 3/40
1354/1354 [==============================] - 213s 157ms/step - loss: 4.7760 - val_loss: 5.6357 - lr: 0.0010
Epoch 4/40
1354/1354 [==============================] - 180s 133ms/step - loss: 4.5980 - val_loss: 5.7343 - lr: 0.0010
Epoch 5/40
1354/1354 [==============================] - 206s 152ms/step - loss: 4.4491 - val_loss: 5.8019 - lr: 0.0010
Epoch 6/40
1354/1354 [==============================] - 220s 163ms/step - loss: 4.3082 - val_loss: 5.8923 - lr: 0.0010
Epoch 7/40
1354/1354 [==============================] - 237s 175ms/step - loss: 4.1871 - val_loss: 5.9825 - lr: 0.0010
Epoch 8/40
1354/1354 [==============================] - 246s 182ms/step - loss: 4.0765 - val_loss: 6.0720 - lr: 0.0010
Epoch 9/40
1354/1354 [==============================] - 244s 180ms/step - loss: 3.9846 - val_loss: 6.1076 - lr: 0.0010
Epoch 10/40
1354/1354 [==============================] - 241s 178ms/step - loss: 3.9003 - val_loss: 6.1772 - lr: 0.0010
Epoch 11/40
1354/1354 [==============================] - 251s 186ms/step - loss: 3.8178 - val_loss: 6.2923 - lr: 0.0010
Epoch 12/40
1354/1354 [==============================] - 228s 169ms/step - loss: 3.7527 - val_loss: 6.3304 - lr: 0.0010
Epoch 13/40
1354/1354 [==============================] - 248s 183ms/step - loss: 3.6894 - val_loss: 6.4129 - lr: 0.0010
Epoch 14/40
1354/1354 [==============================] - 289s 214ms/step - loss: 3.6285 - val_loss: 6.4096 - lr: 0.0010
Epoch 15/40
1354/1354 [==============================] - 271s 200ms/step - loss: 3.5667 - val_loss: 6.4900 - lr: 0.0010
Epoch 16/40
1354/1354 [==============================] - 310s 229ms/step - loss: 3.5191 - val_loss: 6.5465 - lr: 0.0010
Epoch 17/40
1354/1354 [==============================] - 313s 231ms/step - loss: 3.4687 - val_loss: 6.5752 - lr: 0.0010
Epoch 18/40
1354/1354 [==============================] - 307s 227ms/step - loss: 3.4264 - val_loss: 6.6360 - lr: 0.0010
Epoch 19/40
1354/1354 [==============================] - 290s 214ms/step - loss: 3.3725 - val_loss: 6.6482 - lr: 0.0010
Epoch 20/40
1354/1354 [==============================] - 300s 221ms/step - loss: 3.3384 - val_loss: 6.7574 - lr: 0.0010
Epoch 21/40
1354/1354 [==============================] - 310s 229ms/step - loss: 3.3000 - val_loss: 6.7824 - lr: 0.0010
Epoch 22/40
1354/1354 [==============================] - 309s 228ms/step - loss: 3.2521 - val_loss: 6.7838 - lr: 0.0010
Epoch 23/40
1354/1354 [==============================] - 345s 255ms/step - loss: 3.2236 - val_loss: 6.7821 - lr: 0.0010
Epoch 24/40
1354/1354 [==============================] - 377s 279ms/step - loss: 3.1852 - val_loss: 6.8053 - lr: 0.0010
Epoch 25/40
1354/1354 [==============================] - 338s 250ms/step - loss: 3.1603 - val_loss: 6.9187 - lr: 0.0010
Epoch 26/40
1354/1354 [==============================] - 333s 246ms/step - loss: 3.1270 - val_loss: 6.9357 - lr: 0.0010
Epoch 27/40
1354/1354 [==============================] - 360s 266ms/step - loss: 3.1088 - val_loss: 6.9949 - lr: 0.0010
Epoch 28/40
1354/1354 [==============================] - 357s 264ms/step - loss: 3.0751 - val_loss: 6.9508 - lr: 0.0010
Epoch 29/40
1354/1354 [==============================] - 361s 267ms/step - loss: 3.0383 - val_loss: 7.0162 - lr: 0.0010
Epoch 30/40
1354/1354 [==============================] - 379s 280ms/step - loss: 3.0237 - val_loss: 7.0394 - lr: 0.0010
Epoch 31/40
1354/1354 [==============================] - 391s 289ms/step - loss: 2.9949 - val_loss: 6.9890 - lr: 0.0010
Epoch 32/40
1354/1354 [==============================] - 404s 298ms/step - loss: 2.9758 - val_loss: 7.0369 - lr: 0.0010
Epoch 33/40
1354/1354 [==============================] - 376s 278ms/step - loss: 2.9434 - val_loss: 7.0850 - lr: 0.0010
Epoch 34/40
1354/1354 [==============================] - 418s 309ms/step - loss: 2.9254 - val_loss: 7.0887 - lr: 0.0010
Epoch 35/40
1354/1354 [==============================] - 358s 265ms/step - loss: 2.9036 - val_loss: 7.1275 - lr: 0.0010
Epoch 36/40
1354/1354 [==============================] - 358s 265ms/step - loss: 2.8837 - val_loss: 7.1416 - lr: 0.0010
Epoch 37/40
1354/1354 [==============================] - 373s 275ms/step - loss: 2.8679 - val_loss: 7.1568 - lr: 0.0010
Epoch 38/40
1354/1354 [==============================] - 385s 284ms/step - loss: 2.8620 - val_loss: 7.1633 - lr: 0.0010
Epoch 39/40
1354/1354 [==============================] - 382s 282ms/step - loss: 2.8305 - val_loss: 7.2053 - lr: 0.0010
Epoch 40/40
1354/1354 [==============================] - 392s 290ms/step - loss: 2.8148 - val_loss: 7.2181 - lr: 0.0010
```

## 다음 토큰 순위 테스트 로그 (```train.py``` 실행)
```
tokens: ['what', 'was', 'the', 'most', 'number', 'of', 'people', 'you', 'have', 'ever', 'met', 'during', 'a', 'working', 'day', '', '?', '<person-change>']
ID of tokens: [2708, 2681, 2463, 1565, 1637, 1651, 1772, 2782, 1106, 814, 1521, 752, 71, 2750, 640, 0, 57, 53]
ID of tokens: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 2708, 2681, 2463, 1565, 1637, 1651, 1772, 2782, 1106, 814, 1521, 752, 71, 2750, 640, 0, 57, 53]
mini chatgpt model output: [6.2374836e-03 1.0866306e-04 3.5363537e-06 ... 5.4094992e-03 7.7569339e-06
 5.8443625e-06]
first 10 of token arr: ['', '!', '!--', '!unlike', "'billions'", "'court'", "'d", "'m", "'re", "'s"]
rank 0 : ['i', 0.2987077534198761]
rank 1 : ['yes', 0.0817190632224083]
rank 2 : ['no', 0.06997236609458923]
rank 3 : ['it', 0.049224939197301865]
rank 4 : ['yeah', 0.03400596231222153]
rank 5 : ['i\\u2019m', 0.027704738080501556]
rank 6 : ['hmm', 0.027044137939810753]
rank 7 : ['nope', 0.02133043296635151]
rank 8 : ['oh', 0.019310245290398598]
rank 9 : ['nice', 0.018221881240606308]
rank 10 : ['well', 0.017435593530535698]
rank 11 : ['not', 0.01412132941186428]
rank 12 : ['my', 0.012579667381942272]
rank 13 : ['admit', 0.012341052293777466]
rank 14 : ['yea', 0.011254978366196156]
rank 15 : ['the', 0.010231908410787582]
rank 16 : ['arghhh', 0.009504579938948154]
rank 17 : ['slowly', 0.008842614479362965]
rank 18 : ['either', 0.008638895116746426]
rank 19 : ['hi', 0.007381864823400974]

tokens: ['i', 'know', 'him', 'very', 'well', '', '.', '<person-change>', 'is', 'him', 'your', 'friend', '', '?', 'if', 'so', '', ',', 'it', '<person-change>']
ID of tokens: [1204, 1343, 1144, 2643, 2704, 0, 21, 53, 1280, 1144, 2784, 975, 0, 57, 1218, 2220, 0, 16, 1283, 53]
ID of tokens: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 1204, 1343, 1144, 2643, 2704, 0, 21, 53, 1280, 1144, 2784, 975, 0, 57, 1218, 2220, 0, 16, 1283, 53]
mini chatgpt model output: [1.21580646e-03 1.04220374e-03 1.03128390e-04 ... 1.13687514e-04
 2.16827597e-04 9.85123697e-05]
first 10 of token arr: ['', '!', '!--', '!unlike', "'billions'", "'court'", "'d", "'m", "'re", "'s"]
rank 0 : ['i', 0.06089867278933525]
rank 1 : ['hi', 0.04277623072266579]
rank 2 : ['likewise', 0.04143042862415314]
rank 3 : ['yea', 0.0261507797986269]
rank 4 : ['it', 0.0240669846534729]
rank 5 : ['cool', 0.018802452832460403]
rank 6 : ['darn', 0.01819765940308571]
rank 7 : ['oh', 0.01783752255141735]
rank 8 : ['what', 0.01709011197090149]
rank 9 : ['that', 0.017068391665816307]
rank 10 : ['good', 0.016967127099633217]
rank 11 : ['if', 0.015611771494150162]
rank 12 : ['can', 0.014891590923070908]
rank 13 : ['are', 0.014885250478982925]
rank 14 : ['a', 0.014509468339383602]
rank 15 : ['how', 0.01358821801841259]
rank 16 : ['very', 0.013330982066690922]
rank 17 : ['do', 0.011918069794774055]
rank 18 : ['seems', 0.011474553495645523]
rank 19 : ['sounds', 0.011462043970823288]

tokens: ['how', 'can', 'i', 'do', 'for', 'you', '', '?', '<person-change>', 'can', 'you', 'buy', 'me', 'a', 'book', '', '?', '<person-change>']
ID of tokens: [1184, 381, 1204, 713, 951, 2782, 0, 57, 53, 381, 2782, 365, 1485, 71, 311, 0, 57, 53]
ID of tokens: [51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 1184, 381, 1204, 713, 951, 2782, 0, 57, 53, 381, 2782, 365, 1485, 71, 311, 0, 57, 53]
mini chatgpt model output: [7.0006917e-03 7.9933787e-04 7.2961434e-04 ... 5.0627644e-04 1.4363752e-05
 2.3181334e-05]
first 10 of token arr: ['', '!', '!--', '!unlike', "'billions'", "'court'", "'d", "'m", "'re", "'s"]
rank 0 : ['i', 0.11560150980949402]
rank 1 : ['yes', 0.0769299864768982]
rank 2 : ['no', 0.04746954143047333]
rank 3 : ['it', 0.03727545216679573]
rank 4 : ['hi', 0.03049604967236519]
rank 5 : ['right', 0.029437655583024025]
rank 6 : ['yeah', 0.026611367240548134]
rank 7 : ['giants', 0.024289732798933983]
rank 8 : ['maybe', 0.024005664512515068]
rank 9 : ['haha', 0.02304694801568985]
rank 10 : ['that', 0.0221746526658535]
rank 11 : ['oh', 0.021436817944049835]
rank 12 : ['not', 0.02062203548848629]
rank 13 : ['ok', 0.020316753536462784]
rank 14 : ['yea', 0.017908183857798576]
rank 15 : ['thats', 0.017502881586551666]
rank 16 : ['nice', 0.015530657954514027]
rank 17 : ['my', 0.015102026052772999]
rank 18 : ['everything', 0.014467187225818634]
rank 19 : ['cool', 0.013721117749810219]
```

## 대화 테스트 로그 (```test.py``` 실행)
현재 AI 답변은 출력된 token 을 단순히 공백으로 연결한 것이며, 실제 서비스로 배포할 때는 해당 부분을 픽스해야 합니다.

```
input text :
What is your favorite season?
AI output :
i 'm afraid i do n't lose the auras of positive outlook .

input text :
What is the positive outlook you mean?
AI output :
i 'm a little bit . it turns ending ?

input text :
Yes, it turns ending.
AI output :
hi !

input text :
Hello! Nice to meet you!
AI output :
hi !

input text :
Hello!
AI output :
hi !

input text :
How are you?
AI output :
pretty busy . i usual -ly hide it . it 's amazing . it 's a tough decision soon

input text :
Why are you busy then?
AI output :
i slammed onto a lot of bumping into the beginner oven is tallest challenge .

input text :
What do you mean, tallest challenge?
AI output :
mozambique ! ? ?

input text :
What is mozambique?
AI output :
i have n't been a lot . i learned bored spartan food . i think it makes a couple weeks thing to build the correct .

input text :
What is the couple weeks thing?
AI output :
i have n't been to a mentee \u0001f642

input text :
mentee?
AI output :
hi !

input text :
Do you want to build a snowman?
error: 'snowman'
all outputs: i 'm afraid i do n't lose the auras of positive outlook . <person-change> i 'm a little bit . it turns ending ? <person-change> hi ! <person-change> hi ! <person-change> hi ! <person-change> pretty busy . i usual -ly hide it . it 's amazing . it 's a tough decision soon <person-change> i slammed onto a lot of bumping into the beginner oven is tallest challenge . <person-change> mozambique ! ? ? <person-change> i have n't been a lot . i learned bored spartan food . i think it makes a couple weeks thing to build the correct . <person-change> i have n't been to a mentee \u0001f642 <person-change> hi ! <person-change>
```

## 총평
train loss 3.6 부근에서 모델 학습 (fit) 을 종료한 모델들에 비해, train loss 2.8 부근에서 fit이 종료된 현재 모델의 대화 테스트 로그상의 대화가 조금 더 자연스러운 것처럼 보인다.
* valid loss가 계속 증가하는데, 다음 토큰으로 자연스러운 것이 1가지만 있는 것이 아니기 때문에, 가장 적절한 1가지를 예측하는 것은 오차가 있어도 괜찮다. **중요한 건 꺾이지 않는, AI 응답의 자연스러움** 이니까.
* 그러나 train loss를 적어도 2점대 초반 (가급적 1점대 중반 아래로) 까지 줄여서 대화를 조금 더 자연스러워지게 해야 할 듯하다.