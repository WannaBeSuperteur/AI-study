# Deep Q Learning

2024.02.04 작성중

## Deep Q Learning 이란?
**Deep Q Learning** 은 일반적인 Q Learning에서의 Q function의 값을 학습하기 위한 인공신경망을 이용한 **딥 러닝 방식의 학습 방법** 이다.

## Deep Q Learning의 모델 구조
일반적인 Q learning이 현재의 state에 대해서 각 action별로 Q table에서 Q value를 찾아서 그 값이 가장 큰 action을 선택하는 방식이라면, Deep Q Learning의 신경망 구조는 다음과 같다.
* 입력값 : **현재 상태 (state)**
* 출력값 : **각 action에 따른 Q 값들**
  * 출력값은 각 action 당 1개씩 존재하며, 각 출력값은 해당 action을 수행했을 때의 Q value를 의미한다.

## Deep Q Learning의 objective function

## Deep Q Network란?

## Experience Replay 학습 방법
**Experience Replay** 는 Deep Q Learning을 하는 에이전트가 이전의 행동 및 이에 대한 보상 값을 **다시 한번 학습** 하는 것을 의미한다. 이것을 통해 다음과 같은 효과를 얻을 수 있다.
* 에이전트가 데이터를 다시 한번 학습하는 것을 통해 데이터 사용의 효율성을 높인다.
* 에이전트가 학습한 특정 에피소드의 bias가 클 때, 해당 에피소드를 진행하는 step을 순서대로 학습하면 Deep Q Learning이 수렴하기 어려워질 수 있는데, 이를 방지하기 위해서 **여러 개의 episode에 있는 각 step을 랜덤하게 추출하여 재학습** 하는 것이다.
