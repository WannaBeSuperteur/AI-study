## 목차
* [1. AI (인공지능), ML (머신러닝), DL (딥러닝) 의 차이](#1-ai-인공지능-ml-머신러닝-dl-딥러닝-의-차이)
* [2. AI, ML, DL의 예시](#2-ai-ml-dl의-예시)

## 1. AI (인공지능), ML (머신러닝), DL (딥러닝) 의 차이
* **AI (인공지능, Artificial Intelligence)** : 사람의 인지 능력 (학습, 추론 등) 을 컴퓨터 알고리즘을 통해 구현하는 기술
* **ML (머신러닝, 기계학습, Machine Learning)** : 컴퓨터 알고리즘을 통해, 데이터가 주어졌을 때 그 패턴을 학습하여 새로운 데이터에 대해 추론할 수 있게 하는 기술
* **DL (딥 러닝, 심층학습, Deep Learning)** : 사람의 두뇌를 모방한 인공신경망 (Neural Network) 을 통해 머신러닝을 구현하는 것
  * 일반적으로 인공신경망의 hidden layer (은닉층) 가 2개 이상일 때 딥 러닝이라고 부른다.

따라서, 포함 관계는 **AI > ML > DL** 이라고 할 수 있다.

## 2. AI, ML, DL의 예시
* **AI 에는 포함**되지만 ML 에는 포함되지 않는 것
  * 길찾기 알고리즘 (A* 알고리즘 등) : 다익스트라 알고리즘 (Dijkstra's Algorithm) 을 확장하여, 그래프 자료구조의 출발점에서 목표 지점까지의 최단 경로를 탐색하는 알고리즘의 일종
    * 인간의 길찾기 능력을 모방하는 것이지만 데이터를 통해 학습하는 것은 아니다.
* **ML 에는 포함**되지만 DL 에는 포함되지 않는 것
  * Naive Bayes 모델, K-Nearest Neighbor 알고리즘, Decision Tree 알고리즘 등
    * 데이터를 기반으로 학습하는 알고리즘이지만, 인공신경망을 이용하지는 않는다.
* **DL 에 포함**되는 것
  * Convolutional Neural Network (CNN, 합성곱 신경망)
    * CNN의 응용 구조 (ResNet, GoogLeNet, DenseNet, AlexNet 등)
  * Recurrent Neural Network (RNN)