
import json
import pandas as pd

from datasets import DatasetDict, Dataset
from trl import DataCollatorForCompletionOnlyLM

from llm_fine_tuning import LLM_PATH, ANSWER_PREFIX, ANSWER_START_MARK
from llm_fine_tuning import get_llm, train_llm, train_llm_for_tool_call


# NEW_CHAT_TEMPLATE Generated using GPT-5.2 Thinking
NEW_CHAT_TEMPLATE = r"""
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
{{- '<|im_start|>user\n' }}
{{- message['content'] }}
{{- '<|im_end|>\n' }}

    {%- elif message['role'] == 'assistant' %}
{% generation %}
{{- '<|im_start|>assistant' }}
        {%- if message.get('content') %}
{{- '\n' + message['content'] }}
        {%- endif %}
        {%- if message.get('tool_calls') %}
            {%- for tc in message['tool_calls'] %}
                {%- set tcf = tc['function'] if tc.get('function') is defined else tc %}
{{- '\n<tool_call>\n' }}
{{- '{"name": "' + tcf['name'] + '", "arguments": ' + (tcf['arguments'] | tojson) + '}' }}
{{- '\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
{{- '<|im_end|>\n' }}
{% endgeneration %}

    {%- elif message['role'] == 'tool' %}
{{- '<|im_start|>tool\n' }}
        {%- if message.get('content') %}
{{- message['content'] }}
        {%- endif %}
{{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


def generate_llm_trainable_dataset(dataset_df):
    dataset = DatasetDict()
    row_count = len(dataset_df)
    train_row_count = int(0.85 * row_count)

    dataset['train'] = Dataset.from_pandas(dataset_df[:train_row_count][['text']])
    dataset['valid'] = Dataset.from_pandas(dataset_df[train_row_count:][['text']])

    return dataset


def build_row_for_tool_call(row):
    """
    Convert into Tool Call row for tool-calling LLM Dataset.
    Create Date: 2026.02.25

    :param row: original row from Training Dataset (Pandas DataFrame)
    :return: converted json dict
    """

    tool_call_output_info = json.loads(row["tool_call_output"])
    messages = [
        {"role": "user", "content": row["user_input"]},
        {"role": "assistant",
         "tool_calls": [
             {
                 "type": "function",
                 "function": {
                     "name": tool_call_output_info["tool_call"]["name"],
                     "arguments": tool_call_output_info["tool_call"]["arguments"]
                 }
             }
         ]}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate_date",
                "description": "Calculate the date before/after N days from original date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_str": {
                            "type": "string",
                            "description": "original date string, in the format of \"yyyy-mm-dd\""
                        },
                        "days": {
                            "type": "integer",
                            "description": "number of days (positive for after, negative for before)"
                        }
                    },
                    "required": ["date_str", "days"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_day_of_week",
                "description": "Calculate day-of-week of the date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date_str": {
                            "type": "string",
                            "description": "original date string, in the format of \"yyyy-mm-dd\""
                        }
                    },
                    "required": ["date_str"],
                    "additionalProperties": False
                }
            }
        }
    ]

    return {
        "messages": messages,
        "tools": tools
    }


def generate_llm_trainable_dataset_for_tool_call(dataset_df):
    """
    Generate LLM Trainable Dataset for tool calling.
    Create Date: 2026.02.25

    :param dataset_df: Original LLM Training Dataset (Pandas DataFrame)
    :return: converted dataset with tool calling
    """

    dataset = DatasetDict()
    row_count = len(dataset_df)
    train_row_count = int(0.85 * row_count)

    train_df = dataset_df.iloc[:train_row_count].copy()
    valid_df = dataset_df.iloc[train_row_count:].copy()

    train_records = [build_row_for_tool_call(row) for _, row in train_df.iterrows()]
    valid_records = [build_row_for_tool_call(row) for _, row in valid_df.iterrows()]

    dataset["train"] = Dataset.from_list(train_records)
    dataset["valid"] = Dataset.from_list(valid_records)

    return dataset


def train_llm_with_dataset_df(dataset_df, lora_llm, tokenizer, num_train_epochs, save_model_dir, is_tool_call=False):
    """
    Train LLM (Large Language Model) using given Dataset DataFrame.
    Create Date: 2026.02.20
    Last Update Date: 2026.02.25 (handle tool calling)

    :param dataset_df:       Dataset DataFrame
    :param lora_llm:         LLM to fine-tune
    :param tokenizer:        Tokenizer of LLM to fine-tune
    :param num_train_epochs: Train Epoch count
    :param save_model_dir:   Directory to save fine-tuned LLM
    :param is_tool_call:     True for Tool Call, False for just output generation
    """

    if is_tool_call:
        dataset = generate_llm_trainable_dataset_for_tool_call(dataset_df)
    else:
        dataset = generate_llm_trainable_dataset(dataset_df)

    # create data collator
    if is_tool_call:
        train_llm_for_tool_call(lora_llm,
                                dataset,
                                save_model_dir=save_model_dir,
                                num_train_epochs=num_train_epochs,
                                max_length=128)

    else:
        response_template = tokenizer.encode(ANSWER_START_MARK, add_special_tokens=False)
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

        # train LLM
        train_llm(lora_llm,
                  dataset,
                  collator,
                  save_model_dir=save_model_dir,
                  num_train_epochs=num_train_epochs,
                  max_length=128)


def fine_tune_execute_tool_call_llm(dataset_df, lora_llm, tokenizer):
    """
    Train LLM (Large Language Model) for Tool Call.
    Create Date: 2026.02.20
    Last Update Date: 2026.02.25 (handle tool calling)

    :param dataset_df: Dataset DataFrame
    :param lora_llm:   LLM to fine-tune
    :param tokenizer:  Tokenizer of LLM to fine-tune
    """

    train_llm_with_dataset_df(dataset_df,
                              lora_llm,
                              tokenizer,
                              num_train_epochs=5,
                              save_model_dir='execute_tool_call_llm',
                              is_tool_call=True)


def fine_tune_final_output_llm(dataset_df, lora_llm, tokenizer):
    """
    Train LLM (Large Language Model) for Final Output to user.
    Create Date: 2026.02.20
    Last Update Date: 2026.02.21 (fix LLM input format + epoch count)

    :param dataset_df: Dataset DataFrame
    :param lora_llm:   LLM to fine-tune
    :param tokenizer:  Tokenizer of LLM to fine-tune
    """

    dataset_df['text'] = dataset_df.apply(
        lambda x: f"{x['user_input']} -> {x['tool_call_result']} {ANSWER_PREFIX} {ANSWER_START_MARK} {x['final_output']}{tokenizer.eos_token}",
        axis=1
    )

    train_llm_with_dataset_df(dataset_df,
                              lora_llm,
                              tokenizer,
                              num_train_epochs=20,
                              save_model_dir='final_output_llm')


if __name__ == '__main__':

    # Fine-Tuning tool call LLM
    lora_llm, tokenizer = get_llm(LLM_PATH)
    tokenizer.chat_template = NEW_CHAT_TEMPLATE
    dataset_df = pd.read_csv('../toolcall_training_data.csv')
    fine_tune_execute_tool_call_llm(dataset_df, lora_llm, tokenizer)

    # Fine-Tuning final output LLM
#    lora_llm, tokenizer = get_llm(LLM_PATH)
#    fine_tune_final_output_llm(dataset_df, lora_llm, tokenizer)

