## 목차

* [1. MCP (Model Context Protocol) 개요](#1-mcp-model-context-protocol-개요)
  * [1-1. LLM 과 LLM Agent 의 차이](#1-1-llm-과-llm-agent-의-차이) 
* [2. MCP의 구성 요소](#2-mcp의-구성-요소) 
  * [2-1. Host (호스트)](#2-1-host-호스트)
  * [2-2. MCP Client (클라이언트)](#2-2-mcp-client-클라이언트)
  * [2-3. MCP Server (서버)](#2-3-mcp-server-서버)
* [3. MCP의 작동 방법](#3-mcp의-작동-방법)

## 참고 자료

* [MCP (Model Context Protocol) 이 뭐길래? - DEV.DY, 2025.03.20](https://dytis.tistory.com/112)
* [MCP란 무엇인가: LLM Agent 동작 흐름으로 이해하는 MCP - 한컴 기술블로그, 2025.04.28](https://tech.hancom.com/mcp-llm-agent/)

## 1. MCP (Model Context Protocol) 개요

**MCP (Model Context Protocol)** 는 **거대 언어 모델과 외부 데이터 및 시스템이 더 잘 연결되도록 하는 프로토콜** 이다.

* AI 모델과 DB / 시스템 간의 **표준화된 연결 방식** 이라고 할 수 있다.
* **LLM Agent** 를 구현하기 위한 표준 규격이라고 할 수 있다.

### 1-1. LLM 과 LLM Agent 의 차이

* LLM 과 LLM Agent 의 일반적인 차이점은 다음과 같다.

|       | LLM           | LLM Agent                                                |
|-------|---------------|----------------------------------------------------------|
| 목적    | 일회성 언어 처리     | **비교적 복잡한 task** 처리                                      |
| 추가 도구 | -             | **메모리** 기록, **도구 사용** (API 등), 도구 사용 **플래너 (Planner)** 등 |
| 동작 방식 | 단일 입력 → 단일 출력 | ```{입력 → 관련 도구 사용 → 결과 해석 → 출력}``` 등의 절차를 반복 가능          |

## 2. MCP의 구성 요소

### 2-1. Host (호스트)

### 2-2. MCP Client (클라이언트)

### 2-3. MCP Server (서버)

## 3. MCP의 작동 방법