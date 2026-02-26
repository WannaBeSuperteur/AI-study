
from tool_functions import calculate_date_, calculate_day_of_week_

import re
import json
import torch
from typing import Union, ClassVar, Pattern

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.agents import AgentExecutor, AgentOutputParser, create_react_agent
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer, AutoConfig, BitsAndBytesConfig


ORIGINAL_MIDM_LLM_PATH = 'llm_fine_tuning/midm_original_llm'
ANSWER_PREFIX = '(답변 시작)'
LANGCHAIN_ASSISTANT_PREFIX = '<|start_header_id|>assistant<|end_header_id|>\n\n'


# most part of ToolCallTagOutputParser generated using ChatGPT-5.2 Thinking
class ToolCallTagOutputParser(AgentOutputParser):
    """
    LLM output examples:
        <tool_call>
        {"name": "calculate_day_of_week", "arguments": {"date_str": "2024-06-18"}}
        </tool_call>
    """

    TOOL_CALL_PATTERN: ClassVar[Pattern[str]] = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL | re.IGNORECASE
    )

    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        raw = (text or "").strip()

        # parse tool calling
        m = self.TOOL_CALL_PATTERN.search(raw)

        if m:
            payload_str = m.group(1).strip()
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError as e:
                return AgentFinish(return_values={"output": f"[JSONDecodeError] {e}"}, log=raw)

            tool_name = payload.get("name", "")
            tool_args = payload.get("arguments", {}) or {}
            return AgentAction(tool=tool_name, tool_input=tool_args, log=raw)

        # handle fallback (not tool call or error)
        return AgentFinish(return_values={"output": raw}, log=raw)


def load_langchain_llm(llm_path: str):
    """
    Load LLM (Large Language Model) for LangChain.
    Create Date: 2026.02.22

    :param llm_path: Path of LLM
    :return:         LLM for LangChain
    """

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    if tokenizer.pad_token is None:
        print('pad token of tokenizer is None, so add pad token')
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    if tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = '<pad>'

    config = AutoConfig.from_pretrained(ORIGINAL_MIDM_LLM_PATH)
    config.vocab_size = len(tokenizer)  # new vocab size

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='bfloat16'
    )

    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        config=config,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        ignore_mismatched_sizes=True
    )
    llm.resize_token_embeddings(len(tokenizer))
    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.config.eos_token_id = tokenizer.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.6,
        top_p=0.95
    )
    langchain_llm = HuggingFacePipeline(pipeline=pipe)

    return langchain_llm


def run_agent(tool_call_agent_executor, final_output_llm_chat_llm):
    """
    Run LLM Agent.
    Create Date: 2026.02.22
    Last Update Date: 2026.02.26 (tool call 재 구현)

    :param tool_call_agent_executor:  LLM Agent Executor (for Tool Call LLM)
    :param final_output_llm_chat_llm: LangChain LLM to convert Tool Call result to Final Output
    """

    while True:
        user_input = input('\nUSER INPUT:\n')

        # execute tool
        tool_execute_result = tool_call_agent_executor.invoke({'input': user_input})
        tool_execute_result_str = tool_execute_result['intermediate_steps'][-1][1]
        print(f'tool_execute_result : {tool_execute_result_str}')

        # convert to final answer
        final_llm_prompt = ChatPromptTemplate.from_template(
            '{user_input} -> {tool_execute_result}' + f' {ANSWER_PREFIX}'
        )
        print(f'final_llm_prompt : {final_llm_prompt}')

        final_chain = final_llm_prompt | final_output_llm_chat_llm
        final_result = final_chain.invoke({'user_input': user_input, 'tool_execute_result': tool_execute_result_str})
        final_result_content = final_result.content.split(LANGCHAIN_ASSISTANT_PREFIX)[-1]
        print(f'final result : {final_result_content}')


if __name__ == '__main__':

    # BEFORE RUN:

    # 1. edit 4th line of 'llm_fine_tuning/execute_tool_call_llm/adapter_config.json' as below:
    #    - "base_model_name_or_path": "llm_fine_tuning/midm_original_llm"

    # 2. edit 4th line of 'llm_fine_tuning/final_output_llm/adapter_config.json' as below:
    #    - "base_model_name_or_path": "llm_fine_tuning/midm_original_llm"

    execute_tool_call_llm = load_langchain_llm('llm_fine_tuning/execute_tool_call_llm')
    final_output_llm = load_langchain_llm('llm_fine_tuning/final_output_llm')

    # Create and run LLM Agent
    execute_tool_call_chat_llm = ChatHuggingFace(llm=execute_tool_call_llm)
    final_output_llm_chat_llm = ChatHuggingFace(llm=final_output_llm)

    # tool binding
    tools = [calculate_date_, calculate_day_of_week_]
    prompt = PromptTemplate.from_template("tools: {tools}, tool_names: {tool_names} / {input} {agent_scratchpad}")

    # prepare agent executor
    output_parser = ToolCallTagOutputParser()

    tool_call_agent = create_react_agent(
        llm=execute_tool_call_chat_llm,
        tools=tools,
        prompt=prompt,
        output_parser=output_parser
    )

    tool_call_agent_executor = AgentExecutor(
        agent=tool_call_agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=1
    )

    run_agent(tool_call_agent_executor, final_output_llm_chat_llm)
