## 목차

* [1. RAG (Retrieval-Augmented Generation)](#1-rag-retrieval-augmented-generation)
  * [1-1. LangChain의 RAG에서의 고려 요소들](#1-1-langchain의-rag에서의-고려-요소들) 
* [2. Document Loader](#2-document-loader)
* [3. Text Splitter](#3-text-splitter)
* [4. Embedding 알고리즘](#4-embedding-알고리즘)
* [5. Vector Store](#5-vector-store)
  * [5-1. Vector Store 의 벡터 유사도 기준](#5-1-vector-store-의-벡터-유사도-기준) 

## 1. RAG (Retrieval-Augmented Generation)

* **RAG (Retrieval-Augmented Generation), 검색 증강 생성** 은 기존의 언어 모델에 **검색 가능한 DB** 를 연동하여, LLM이 해당 DB로부터 적절한 정보를 검색하여 **더 정확하고 풍부한 정보를 기반으로 답변을 생성** 하게 하는 것이다.
* 참고
  * [LLM 기초 - RAG](../AI%20Basics/LLM%20Basics/LLM_기초_RAG.md)
  * [LLM 기초 - LangChain - LangChain의 구성 요소](../AI%20Basics/LLM%20Basics/LLM_기초_Langchain.md#2-langchain-의-구성-요소)

![image](../AI%20Basics/LLM%20Basics/images/RAG_1.PNG)

### 1-1. LangChain의 RAG에서의 고려 요소들

LangChain에서 RAG을 사용할 때 고려해야 하는 요소들은 다음과 같다.

| 고려 대상 요소        | 설명                                                    |
|-----------------|-------------------------------------------------------|
| Document Loader | 문서 호출 및 처리 작업에 필요                                     |
| Text Splitter   | 긴 문서를 **chunk 로 분리** 하기 위해 사용하는 도구                    |
| 임베딩 알고리즘        | **문서 분류 및 유사도 계산** 등을 위해, **텍스트 데이터를 숫자 벡터로 변환** 하는 것 |
| Vector Store    | 임베딩 벡터를 **효율적으로 저장, 검색** 가능한 시스템/DB                   |

## 2. Document Loader

LangChain 에서 제공하는 Document Loader 의 종류는 다음과 같다.

| Document Loader 종류              | 설명                                                                                                                                                                                                                                             |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 웹 문서 (```WebBaseLoader```)      | 웹 페이지 (URL) 에서 문서 로딩                                                                                                                                                                                                                           |
| 텍스트 문서 (```TextLoader```)       | 저장된 텍스트 파일 (*.txt 등) 에서 텍스트 데이터 로딩                                                                                                                                                                                                             |
| 디렉토리 폴더 (```DirectoryLoader```) | 특정 디렉토리 내에 저장된 모든 문서를 로딩                                                                                                                                                                                                                       |
| csv 문서 (```CSVLoader```)        | csv 파일로부터 데이터 추출                                                                                                                                                                                                                               |
| PDF 문서                          | - 페이지별 로딩 (```PyPDFLoader```)<br>- 형식이 없는 PDF 문서 로딩 (```UnstructuredPDFLoader```)<br>- 메타데이터 상세 추출 (```PyMuPDFLoader```)<br>온라인 PDF 문서 로딩 (```OnlinePDFLoader```) - **arXiv 논문 등 로딩 가능**<br>- 특정 폴더의 모든 PDF 문서 로딩 (```PyPDFDirectoryLoader```) |

## 3. Text Splitter

LangChain 에서 제공하는 Text Splitter 의 종류는 다음과 같다.

* Text Splitter는 **긴 문서를 chunk 로 분리** 하기 위해 사용된다.

| Text Splitter 종류                     | 설명                                                               |
|--------------------------------------|------------------------------------------------------------------|
| ```CharacterTextSplitter```          | 주어진 텍스트를 문자 단위로 분할<br>- 분할 기준 문자열, 분할된 부분 최대 길이 등 설정 가능          |
| ```RecursiveCharacterTextSplitter``` | 텍스트의 **재귀적 분할** 을 통해, 의미적 관련이 있는 텍스트 조각이 같이 있게 함                 |
| 토큰 수 기준 텍스트 분할 (tokenizer 사용)        | **토큰 개수를 기준으로 텍스트를 chunking** 하여, **거대 언어 모델의 token 개수 한계** 에 대응 |

## 4. Embedding 알고리즘

LangChain에서 제공하는 임베딩 기법은 다음과 같다.

* 임베딩 기법
  * OpenAIEmbeddings
  * HuggingFaceEmbeddings
  * GoogleGenerativeAIEmbeddings
* 참고
  * [LLM 기초 - RAG - RAG을 위한 임베딩 기법](../AI%20Basics/LLM%20Basics/LLM_기초_RAG.md#4-rag-을-위한-임베딩-기법) 

임베딩은 **문서 분류 (카테고리 분류), 의미 검색 (유사한 문서 탐색), 텍스트 간 유사도 계산** 등에 사용된다.

## 5. Vector Store

**Vector Store (벡터 저장소)** 는 임베딩 벡터들을 효율적으로 저장 및 검색하는 시스템 또는 DB이다.

* 벡터 저장소의 핵심 역할은 **대규모 벡터 데이터셋에서 유사한 데이터를 빠르게 찾는 것** 이다.

대표적인 Vector Store인 **Chroma** 와 **FAISS** 의 특징은 다음과 같다.

| 특징         | Chroma DB                              | FAISS                               |
|------------|----------------------------------------|-------------------------------------|
| 개요 (핵심 기능) | **오픈소스** 소프트웨어 / 쉬운 **LLM 기반 앱 구축** 지원 | **대규모 벡터 데이터셋** 에서의 빠르고 효율적인 유사도 검색 |
| 영구 저장      | 디스크 사용하여 **영구 저장 가능**                  | 영구 저장되지 않음                          |

### 5-1. Vector Store 의 벡터 유사도 기준

Vector Store 에서 사용하는 벡터 간 유사도 기준은 다음과 같다.

| L2                                  | ip (내적, inner product)        | cosine                     |
|-------------------------------------|-------------------------------|----------------------------|
| 유클리디안 거리 기반<br>(두 벡터의 **유클리디안 거리**) | 내적 기반<br>(두 벡터의 **방향성 + 크기**) | 코사인 유사도<br>(두 벡터의 **방향성**) |
