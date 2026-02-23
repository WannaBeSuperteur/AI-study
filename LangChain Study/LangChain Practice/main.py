from langchain_core.prompts import ChatPromptTemplate

from tool_functions import calculate_date_, calculate_day_of_week_, calculate_date, calculate_day_of_week
import torch
import json

from langchain.agents import create_agent
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer, AutoConfig, BitsAndBytesConfig


ORIGINAL_MIDM_LLM_PATH = 'llm_fine_tuning/midm_original_llm'
ANSWER_PREFIX = '(답변 시작)'
LANGCHAIN_ASSISTANT_PREFIX = '<|start_header_id|>assistant<|end_header_id|>\n\n'


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


def run_agent(agent, final_output_llm_chat_llm):
    """
    Run LLM Agent.
    Create Date: 2026.02.22
    Last Update Date: 2026.02.23 (tool call 재 구현)

    :param agent:                     LLM Agent to run
    :param final_output_llm_chat_llm: LangChain LLM to convert Tool Call result to Final Output
    """

    tool_map = {}
#    for tool in tools_original_functions:
#        tool_map[tool.__name__ + '_'] = tool

    while True:
        user_input = input('\nUSER INPUT:\n')

        # execute tool
        tool_result = agent.invoke({
            "messages": [{'role': 'user', 'content': user_input + f' {ANSWER_PREFIX}'}]
        })
        tool_result_msg = tool_result["messages"][-1]
        tool_result_content = tool_result_msg.content.split(LANGCHAIN_ASSISTANT_PREFIX)[-1]
        tool_result_content_json = json.loads(tool_result_content)

        try:
            tool_call = tool_result_content_json["tool_call"]
            tool_name = tool_call.get("name")
            arg_dict = tool_call.get("arguments")
            tool_execute_result = tool_map[tool_name](**arg_dict)

            print(f'tool execution result : {tool_execute_result}')

        except Exception as e:
            print(e)
            tool_execute_result = '도구 호출 실패'

        # convert to final answer
        final_llm_prompt = ChatPromptTemplate.from_template(
            '{user_input} -> {tool_execute_result}' + f' {ANSWER_PREFIX}'
        )
        print(f'final_llm_prompt : {final_llm_prompt}')

        final_chain = final_llm_prompt | final_output_llm_chat_llm
        final_result = final_chain.invoke({'user_input': user_input, 'tool_execute_result': tool_execute_result})
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

    tools = [calculate_date_, calculate_day_of_week_]

    agent = create_agent(
        model=execute_tool_call_chat_llm,
        tools=tools
    )

    run_agent(agent, final_output_llm_chat_llm)
