
from langchain.agents import create_agent
import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, pipeline


def load_langchain_llm(llm_path: str):
    """
    Load LLM (Large Language Model) for LangChain.
    Create Date: 2026.02.22

    :param llm_path: Path of LLM
    :return:         LLM for LangChain
    """

    llm = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoModelForCausalLM.from_pretrained(llm_path)

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


if __name__ == '__main__':

    # BEFORE RUN:

    # 1. edit 4th line of 'llm_fine_tuning/execute_tool_call_llm/adapter_config.json' as below:
    #    - "base_model_name_or_path": "llm_fine_tuning/midm_original_llm"

    # 2. edit 4th line of 'llm_fine_tuning/final_output_llm/adapter_config.json' as below:
    #    - "base_model_name_or_path": "llm_fine_tuning/midm_original_llm"

    execute_tool_call_llm = load_langchain_llm('llm_fine_tuning/execute_tool_call_llm')
    final_output_llm = load_langchain_llm('llm_fine_tuning/final_output_llm')
