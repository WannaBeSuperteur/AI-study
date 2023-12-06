# Deep learning-based optimal placement of a mobile HAP for common throughput maximization in wireless powered communication networks
다음 논문에서 쓰인 기술에 대해 기본 원리 및 사용 이유 파악
* **[Deep learning-based optimal placement of a mobile HAP for common throughput maximization in wireless powered communication networks](https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-021-02051-w)**

## Adam Optimizer [Paper](https://arxiv.org/pdf/1412.6980.pdf)
**Adam (adaptive moment estimation)**
참고 : [딥러닝 기초 - Optimizer](https://github.com/WannaBeSuperteur/AI-study/blob/main/AI%20Basics/Deep%20Learning%20Basics/%EB%94%A5%EB%9F%AC%EB%8B%9D_%EA%B8%B0%EC%B4%88_Optimizer.md)

**Adam Optimizer의 기본 아이디어**
* 모멘텀 (Momentum) : 기본적인 Gradient Descent를 개선하여, 딥러닝 학습 시 **gradient에 대한 지수가중평균** 을 이용하여 일종의 **마찰력** 을 적용하는 방법이다.
  * 이를 통해 gradient를 구성하는 원소들 중 진동하는 원소에 대해서는 그 양을 줄이고, 빠르게 이동하는 원소에 대해서는 그 이동 속도를 향상시킨다.
* RMSProp (Root Mean Square Prop) : 모멘텀과 같이 Gradient Descent의 진동 문제를 해결하는 방법이다. 모멘텀이 gradient의 지수가중평균을 이용한다면, RMSProp은 **gradient의 제곱** 의 지수가중평균을 이용한다.
* Adam : **모멘텀과 RMSProp 방법을 같이 사용** 하여, 각 알고리즘이 Gradient Descent를 개선시키는 효과를 동시에 볼 수 있다.

**유사한 알고리즘**
* Adagrad : 각 매개변수의 learning rate를 서로 다르게 한다.
* Nadam : RMSProp과 NAG (Nesterov Accelerated Gradient) 를 동시에 적용한다.