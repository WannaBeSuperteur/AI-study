## 목차

* [1. RAG (Retrieval-Augmented Generation)](#1-rag-retrieval-augmented-generation)
  * [1-1. LangChain의 RAG에서의 고려 요소들](#1-1-langchain의-rag에서의-고려-요소들) 
* [2. Document Loader](#2-document-loader)
* [3. Text Splitter](#3-text-splitter)
* [4. Embedding 알고리즘](#4-embedding-알고리즘)
* [5. Vector Store](#5-vector-store)

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

## 3. Text Splitter

## 4. Embedding 알고리즘

## 5. Vector Store