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
* epochs 40 -> 160 으로 테스트 시도
* RMSProp Loss 적용 테스트 시도 (valid loss뿐만 아니라 train loss 마저 높은 상황에서도 실제 AI Chatbot 과의 채팅이 자연스럽게 이루어질 수 있을지도?)

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