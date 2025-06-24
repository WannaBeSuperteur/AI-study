## 목차

* [1. PyTorch 의 state_dict 란?](#1-pytorch-의-statedict-란)
* [2. 모델에 가중치 로딩하기](#2-모델에-가중치-로딩하기)
* [3. state dict 저장하기](#3-state-dict-저장하기)
* [4. state dict 의 layer 별 shape 및 학습 가능 여부 확인하기](#4-state-dict-의-layer-별-shape-및-학습-가능-여부-확인하기)
* [5. 참고하면 좋은 문서](#5-참고하면-좋은-문서)

## 1. PyTorch 의 state_dict 란?

PyTorch 에서 ```state_dict``` 는 **모델의 가중치 및 bias 를 저장하는 일종의 dictionary** 를 의미한다.

* ```state_dict``` 의 형태는 다음과 같다.

```python
OrderedDict([('{layer_name}', tensor([...])), ...])
```

여기서 각 부분에 대한 설명은 다음과 같다.

| 부분                   | 설명                                           |
|----------------------|----------------------------------------------|
| ```'{layer_name}'``` | state_dict 의 해당 부분이 가리키는 레이어 이름              |
| ```tensor([...])```  | state_dict 의 해당 부분이 가리키는 레이어의 PyTorch Tensor |

## 2. 모델에 가중치 로딩하기

PyTorch 모델에 ```state_dict``` 로부터 가중치를 로딩하려면 다음과 같이 하면 된다.

* 이는 [Transfer Learning (전이학습)](딥러닝_기초_Transfer_Learning.md) 에서 [Pre-trained](딥러닝_기초_Transfer_Learning.md#3-1-사전-학습-pre-training) Model 을 로딩하여 [Fine-Tuning](딥러닝_기초_Transfer_Learning.md#3-2-미세-조정-fine-tuning) 하는 등의 상황에 많이 사용된다.

```python
import torch

example_neural_net = ExampleNN()  # ExampleNN 이라는 Class 가 class ExampleNN(nn.Module) 로 정의되어 있어야 함

# state dict 가져오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load('{model_file_name}.pth', map_location=device)

# example_neural_net 모델에 state dict 를 로딩
example_neural_net.load_state_dict(state_dict)
```

## 3. state dict 저장하기

PyTorch 모델을 학습한 후, **학습된 모델의 ```state_dict``` 를 저장** 하려면 다음과 같이 하면 된다.

* 이는 모델을 저장한 후 나중에 추론 (inference) 목적으로 사용할 때 많이 쓰인다.

```python
torch.save(trained_model.state_dict(), '{model_file_name}.pth')
```

이때, **다음과 같이 하는 것은 ```torch.save``` 과정에서 오류가 발생하지는 않지만 나중에 해당 모델을 사용하기 위한 ```torch.load``` 과정에서 오류가 발생하는 잘못된 방법** 이다.

```python
# wrong way ('state_dict()', not 'state_dict')
torch.save(trained_model.state_dict, '{model_file_name}.pth')
```

## 4. state dict 의 layer 별 shape 및 학습 가능 여부 확인하기

```state_dict``` 에서 각 layer 에 대해 tensor 의 shape 및 학습 가능 여부를 확인하려면 다음과 같이 하면 된다.

* 방법 요약

| 방법                              | 필요한 것                             | 알 수 있는 것                           |
|---------------------------------|-----------------------------------|------------------------------------|
| ```state_dict``` 이용             | 모델의 ```state_dict``` (전체 모델은 불필요) | 각 레이어 이름, tensor 의 shape           |
| 모델의 ```named_parameters()``` 이용 | 전체 모델 필요                          | 각 레이어 이름, tensor 의 shape, 학습 가능 여부 |

* 방법 1 (```state_dict``` 이용)

```python
for layer_name, tensor in state_dict.items():
    print(f'layer_name = {layer_name}, tensor shape = {tensor.shape}')
```

```python
# 실행 결과 예시

layer_name = fc1.0.weight, tensor shape = torch.Size([512, 6])
layer_name = fc1.0.bias, tensor shape = torch.Size([512])
layer_name = fc2.0.weight, tensor shape = torch.Size([128, 512])
layer_name = fc2.0.bias, tensor shape = torch.Size([128])
layer_name = fc_final.weight, tensor shape = torch.Size([1, 128])
layer_name = fc_final.bias, tensor shape = torch.Size([1])
```

* 방법 2 (model 의 ```named_parameters()``` 이용)

```python
for name, param in model.named_parameters():
    print(f'layer name = {name}, shape = {param.shape}, trainable = {param.requires_grad}')
```

```python
# 실행 결과 예시 ('trainable = True' 이면 학습 가능, 'trainable = False' 이면 Frozen)

layer name = fc1.0.weight, shape = torch.Size([512, 6]), trainable = True
layer name = fc1.0.bias, shape = torch.Size([512]), trainable = True
layer name = fc2.0.weight, shape = torch.Size([128, 512]), trainable = True
layer name = fc2.0.bias, shape = torch.Size([128]), trainable = True
layer name = fc_final.weight, shape = torch.Size([1, 128]), trainable = True
layer name = fc_final.bias, shape = torch.Size([1]), trainable = True
```

## 5. 참고하면 좋은 문서

* [PyTorch 레이어 가중치 및 출력 확인 방법](딥러닝_실무_PyTorch_레이어_가중치_및_출력_확인.md)
* [PyTorch 모델 구조 시각화 방법](딥러닝_실무_PyTorch_모델_구조_시각화.md)
