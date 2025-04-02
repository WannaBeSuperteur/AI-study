## 개요

* ```glass.py```
  * Vision Anomaly Detection 모델인 GLASS 의 구현 코드
* ```tinyvit.py```
  * Vision Classification 모델인 TinyViT-21M-512-distill 의 구현 코드
  * ```timm``` 라이브러리를 이용하여 간단히 모델을 불러오는 방법으로 [구현할 수 있음](https://huggingface.co/timm/tiny_vit_21m_512.dist_in22k_ft_in1k)
* ```pytorch_grad_cam.py```
  * XAI 모델인 pytorch-grad-cam 의 구현 코드

## 실행 방법

```
python tinyvit.py
python glass.py
python pytorch_grad_cam.py
```