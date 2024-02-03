# Actor Critic, A2C, A3C

2024.02.03 작성중

## 가치함수와 정책함수
가치함수와 정책함수는 각각 다음과 같은 개념이다.
* **가치함수 (value function)** : 주어진 state (상태) 에서의 reward의 기댓값을 추정하기 위한 함수이다.
* **정책함수 (policy function)** : 주어진 state에서의 에이전트의 행동을 그 행동에 대한 확률분포를 이용하여 결정하기 위한 함수이다.
  * 확률분포상의 확률이 높을수록 해당 행동을 통해 가장 높은 보상을 받을 가능성이 크다.

## Actor Critic 이란?
**Actor-Critic** 은 강화학습의 알고리즘 중 하나로, 다음과 같이 **Actor** 와 **Critic** 으로 구성되어 있다.
* **Actor** : 정책함수를 학습하는 역할을 한다. 주어진 상태에서 실행 가능한 행동들에 대한 reward에 기반하여 행동을 최종적으로 선택한다.
* **Critic** : 가치함수를 학습하는 역할을 한다. 즉 에이전트가 최종적으로 선택한 행동의 가치를 평가하여 업데이트한다.

## A2C (Advantage Actor Critic)
**A2C (Advantage Actor Critic)** 는 가치 함수의 값 $V(s)$ 와 비교하여 계산한 **Advantage** 라는 값을 이용한 알고리즘이다.

## A3C (Asynchronous Advantage Actor Critic)
**A3C (Asynchronous Advantage Actor Critic)** 에서는 **비동기 (Asynchronous)** 의 개념을 이용한다. 즉 여러 개의 **에이전트들이 신경망을 공유하고 비동기적으로 업데이트** 하며 학습하는 방식의 알고리즘이다. 이때 각 에이전트들은 **A2C 알고리즘** 을 사용한다.