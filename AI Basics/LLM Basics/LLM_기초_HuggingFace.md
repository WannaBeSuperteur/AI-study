## 목차

* [1. HuggingFace](#1-huggingface)
* [2. HuggingFace 의 기능](#2-huggingface-의-기능)
  * [2-1. Models 메뉴](#2-1-models-메뉴)
  * [2-2. Datasets 메뉴](#2-2-datasets-메뉴)

## 1. HuggingFace

**HuggingFace (허깅페이스)** 는 여러 가지 Pre-train 된 거대 언어 모델을 탐색하고 사용할 수 있는 종합 플랫폼 웹사이트이다. 언어 모델을 학습시키기 위한 데이터셋을 탐색할 수도 있다.

2016년 뉴욕에서 설립된 스타트업 'HuggingFace'에서 운영하고 있다.

## 2. HuggingFace 의 기능

**HuggingFace 의 대표적인 기능** 은 다음과 같다.

| 기능 (메뉴)  | 설명                                                                           |
|----------|------------------------------------------------------------------------------|
| Models   | Pre-train 된 거대 언어 모델을 탐색 및 사용할 수 있음                                          |
| Datasets | 언어 모델을 학습 (기존 거대 언어 모델을 [Fine-tuning](LLM_기초_Fine_Tuning.md) 시키는 데 사용할 수 있음) |

### 2-1. Models 메뉴

**Models** 메뉴는 **다양한 거대 언어 모델을 탐색하고, 다운로드하여 사용** 할 수 있는 공간이다. 언어 모델 분야 말고도 Vision 등 다양한 분야의 모델을 제공하고 있다.

이들 중 거대 언어 모델 또는 [멀티모달 모델](../../Others/Others_Multi%20Modal%20AI.md) 과 관련된 세부 메뉴 (sub-menu) 는 다음과 같다. (2025년 3월 6일 기준)

* 거대 언어 모델

| 세부 메뉴                                                                                                 | 설명                                                      | 주요 모델                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Translation](https://huggingface.co/models?pipeline_tag=translation&sort=trending)                   | 번역 관련 LLM 및 언어 모델                                       |                                                                                                                                                                                                      |
| [Text Generation](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)           | 텍스트를 생성하는 언어 모델                                         | [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1), [LLAMA-3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) 등 |
| [Text2Text Generation](https://huggingface.co/models?pipeline_tag=text2text-generation&sort=trending) | 텍스트를 생성하는 언어 모델<br>(입력과 출력 내용이 서로 관계있는 경우 - 요약, 질의응답 등) |                                                                                                                                                                                                      |

* 멀티모달 AI

| 세부 메뉴                                                                                                               | 설명                                       | 주요 모델                                                           |
|---------------------------------------------------------------------------------------------------------------------|------------------------------------------|-----------------------------------------------------------------|
| [Audio-Text-to-Text](https://huggingface.co/models?pipeline_tag=audio-text-to-text&sort=trending)                   | 음성 자료와 텍스트를 입력으로 받아서, 텍스트를 출력            | [Qwen2-Audio-7B](https://huggingface.co/Qwen/Qwen2-Audio-7B)    |
| [Image-Text-to-Text](https://huggingface.co/models?pipeline_tag=image-text-to-text&sort=trending)                   | 이미지와 텍스트를 입력, 텍스트를 출력<br>(비전 언어 모델, VLM) | [DeepSeek-VL2](https://huggingface.co/deepseek-ai/deepseek-vl2) |
| [Visual Question Answering](https://huggingface.co/models?pipeline_tag=visual-question-answering&sort=trending)     | 이미지 및 그 이미지와 관련된 질문 입력, 답변 출력            |                                                                 |
| [Document Question Answering](https://huggingface.co/models?pipeline_tag=document-question-answering&sort=trending) | 문서 내용 및 그 문서와 질문 입력, 답변 출력               |                                                                 |
| [Video-Text-to-Text](https://huggingface.co/models?pipeline_tag=video-text-to-text&sort=trending)                   | 동영상과 텍스트를 입력, 텍스트를 출력                    |                                                                 |
| [Visual Document Retrieval](https://huggingface.co/models?pipeline_tag=visual-document-retrieval&sort=trending)     | VLM 을 이용하여 시각 자료가 있는 문서를 해석              |                                                                 |
| [Any-to-Any](https://huggingface.co/models?pipeline_tag=any-to-any&sort=trending)                                   | 텍스트, 이미지, 동영상 등 다양한 데이터 형식에 대한 입출력 가능    | [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) |

### 2-2. Datasets 메뉴

**Datasets** 메뉴는 거대 언어 모델을 학습 (Fine-tuning 등) 시킬 수 있는 **다양한 데이터셋을 소개하는 공간** 이다. 텍스트, 이미지, Time-Series (시계열), Video (동영상) 등 다양한 형식의 데이터를 제공한다.

이들 중 거대 언어 모델을 학습시킬 수 있는 텍스트 데이터셋은 [해당 메뉴](https://huggingface.co/datasets?modality=modality:text&sort=trending) 에서 탐색 및 다운로드할 수 있다.