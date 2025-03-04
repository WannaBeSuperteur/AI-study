## 목차

* [1. TensorFlow 개요](#1-tensorflow)
  * [1-1. 모델 학습 기본 코드](#1-1-모델-학습-기본-코드)
  * [1-2. Keras](#1-2-keras)
* [2. PyTorch 개요](#2-pytorch)
  * [2-1. 모델 학습 기본 코드](#2-1-모델-학습-기본-코드)
  * [2-2. PyTorch Lightning](#2-2-pytorch-lightning)
* [3. TensorFlow vs. PyTorch](#3-tensorflow-vs-pytorch)

## 1. TensorFlow

**TensorFlow (텐서플로우)** 는 구글에서 개발하여 2015년에 공개한 딥러닝 프레임워크이다.

* 기본적으로는 모델을 **정적 계산 그래프 (static computation graph)** 를 통해서 먼저 정의하고, 이후 학습 및 테스트에 사용하는 방식이다.
* TensorFlow 2.0 부터는 'Eager Execution' 을 통해 모델의 그래프를 동적으로 생성할 수 있다.

TensorFlow 는 Google 에서 개발한 프레임워크로, Android OS 와의 호환이 잘 되어서 안드로이드 앱 등에 쉽게 적용할 수 있다.
  * Android 앱 서비스에 모델을 직접 적용하고 싶은 서비스 기업에서 사용하기도 한다.

### 1-1. 모델 학습 기본 코드

* 참고 : https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko

```python
import tensorflow as tf
import numpy as np
```

```python
# 데이터셋 로딩
# 입력 데이터가 픽셀 값이므로 이를 255.0 으로 나누어 먼저 정규화해야 함

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

```python
# y_train, y_test 를 one hot vector 화

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]
```

```python
# 모델 정의

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

```python
# 모델 컴파일

model.compile(optimizer='adamw',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

```python
# 모델 학습

model.fit(x_train, y_train, epochs=10)
```

```python
# 모델 성능 테스트

_, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'accuracy : {accuracy:.4f}')
```

```python
# 모델 확률 출력
# 최종 layer 에 이미 Softmax 가 적용되었으므로 추가 변환 불필요

np.set_printoptions(suppress=True, precision=8, linewidth=160)
prediction = model(x_test[:20])
prediction
```

```python
# 확률의 합이 1이 되는지 확인

np.sum(prediction, axis=1)
```

### 1-2. Keras

**Keras (케라스)** 는 TensorFlow 를 통한 딥러닝 모델 개발이 다소 복잡하다는 점을 해결하기 위해 개발된 딥러닝 라이브러리이다. 그 특징은 다음과 같다.

* 간단한 API 를 통해 사용자가 별도로 설정해야 하는 최소화
* 버그 및 오류 발생 시 이에 대한 피드백 제공
* TensorFlow 와의 높은 호환성

## 2. PyTorch

**PyTorch (파이토치)** 는 Facebook 에서 2016년에 공개한 딥러닝 프레임워크이다. 그 특징은 다음과 같다.

* Python 의 코딩 방식과 유사한 간결한 코드 작성 가능
* 딥러닝 모델의 계산 그래프를 동적으로 생성

PyTorch 의 등장 이후, **최근에는 업계에서 TensorFlow 보다는 PyTorch 를 많이 사용하는** 추세이다. 단 모델의 배포가 다소 어렵기 때문에 TensorFlow 보다는 비교적 연구 목적으로 많이 사용된다.

### 2-1. 모델 학습 기본 코드

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
```

```python
# 데이터셋 로딩 및 정규화

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
```

```python
# 모델 정의

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)  # 확률값 출력을 위해 필요

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
```

```python
# 모델 생성 및 컴파일

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

model.loss_function = nn.CrossEntropyLoss()
model.optimizer = optim.AdamW(model.parameters())
```

```python
# 모델 학습

num_epochs = 10

for epoch in range(num_epochs):

    model.train()  # 모델을 train mode (loss back-prop 이 가능) 로 전환

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # (중요) Loss 역전파 시 기존에 저장된 값이 있으면 이로 인해 역전파가 올바르게 진행되지 않는다.
        # 이를 해결하기 위해, 기존에 저장된 값을 model.optimizer.zero_grad() 를 통해 초기화한다.
        model.optimizer.zero_grad()

        outputs = model(images)
        loss = model.loss_function(outputs, labels)

        loss.backward()  # 역전파 실시
        model.optimizer.step()  # optimizer 의 매개변수 조정

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

```python
# 테스트셋을 이용한 모델 테스트

model.eval()  # 모델을 eval mode 로 전환

correct = 0
total = 0

with torch.no_grad():  # loss back-prop (Gradient 연산) 없이 모델 실행
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')
```

```python
# 모델 확률 출력

with torch.no_grad():
    test_samples, _ = next(iter(test_loader))
    test_samples = test_samples.to(device)
    output_probs = model(test_samples[:20])

    print(output_probs.cpu().numpy())
```

```python
# 확률의 합이 1인지 확인

sum_probabilities = output_probs.sum(dim=1)

print(sum_probabilities)
```

**중요 사항**

* 학습 중 ```model.optimizer.zero_grad()``` 사용 이유
  * Loss 를 역전파할 때 모델 학습을 통해 기존에 저장된 값이 있으면, 이 저장된 값의 영향으로 역전파가 올바르게 진행되지 않음
  * 따라서, **기존에 저장된 이 값을 초기화**하기 위해 필요

### 2-2. PyTorch Lightning

**PyTorch Lightning** 은 PyTorch 에 대해 **보다 고수준의 사용자 인터페이스** 를 제공하는 라이브러리이다.

* PyTorch 를 이용한 학습 프로세스가 복잡해질 경우 (Single-Label vs. Multi-Label 구분, GPU 분산 학습 등) 복잡해지는 코드에 대한 **더욱 높은 수준의 추상화** 를 제공한다.
* 즉, **기존 PyTorch 보다도 코드를 더욱 간결하게** 작성할 수 있다.

**PyTorch Lightning 의 장단점**

* 장점
  * **코드의 간결성이 더욱 늘어난다. (사실상 핵심 장점)**
* 단점
  * 고수준의 추상화로 인한 디버깅의 어려움
    * 학습 과정의 특정 단계에서 호출되는 함수 (예: ```training_step(self, ...)``` 함수) 에 대한 Traceback 분석의 어려움 등 
  * 자유도가 떨어질 수 있음
    * PyTorch Lightning 의 학습 프로세스 구조를 우리가 제공하려는 서비스의 학습 프로세스에 적용하기 어려운 경우 

**PyTorch Lightning 의 모델 클래스 포맷**

* PyTorch Lightning 을 이용하여 정의하는 모델 클래스의 포맷은 다음과 같다.

```python
import pytorch_lightning as pl

class CustomModel(pl.LightningModule):

    # 모델 정의
    def __init__(self, ...):
        super(CustomModel, self).__init__()
        ...  
        
    # 모델의 forward propagation 정의
    def forward(self, x):
        return self.model(x) 
    
    # 모델의 batch 단위 학습 프로세스 정의
    # Loss 를 Return 함으로써 모델 back-prop 실시
    def training_step(self, batch, batch_idx):
        ...
        return loss
    
    # 모델의 batch 단위 validation 프로세스 정의
    def validation_step(self, batch, batch_idx):
        ...
    
    # 모델의 batch 단위 test 프로세스 정의
    def test_step(self, batch, batch_idx):
        ...

    # Optimizer 정의
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.001)
```

## 3. TensorFlow vs. PyTorch

TensorFlow 와 PyTorch 의 주요 특징을 비교하면 다음과 같다. 결론적으로 다음과 같다.

* 코드 간결성 등으로 인해 **대부분의 경우에는 PyTorch 가 보다 적합** 하다.
* 모델 자체를 Android 앱 등을 통해 서비스하고자 한다면 TensorFlow 가 보다 적합할 수도 있다.

| 구분           | TensorFlow                                                                | PyTorch                                                                                                           |
|--------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| 출시           | 2015년, Google                                                             | 2016년, Facebook                                                                                                   |
| 현재 업계 인기도    | 비교적 낮음                                                                    | **비교적 높음**                                                                                                        |
| 코드 간결성       | 비교적 낮음                                                                    | **비교적 높음**                                                                                                        |
| 모델 배포 편의성    | **비교적 높음**                                                                | 비교적 낮음                                                                                                            |
| 모델 계산 그래프 생성 | 기본적으로 정적 그래프<br>(v2.0 부터는 동적 그래프 가능)                                      | 동적 그래프 생성                                                                                                         |
| 모델 저장 및 불러오기 | 저장된 모델 파일을 불러올 때,<br>- Keras 를 통해 **모델 Class 를 따로 새로 정의할 필요 없이** 바로 이용 가능 | 저장된 모델 파일을 불러올 때,<br>- 모델 파일은 **가중치 목록 (```state_dict```)**<br>- **해당 모델의 구조를 Class 로 미리 정의** 한 후 그 Class에 불러와야 함 |
| 관련 라이브러리     | Keras                                                                     | PyTorch Lightning                                                                                                 |