
import pandas as pd
from datasets import DatasetDict, Dataset
from trl import DataCollatorForCompletionOnlyLM

from llm_fine_tuning import LLM_PATH, ANSWER_PREFIX, ANSWER_START_MARK
from llm_fine_tuning import get_llm, train_llm


def generate_llm_trainable_dataset(dataset_df):
    dataset = DatasetDict()
    row_count = len(dataset_df)
    train_row_count = int(0.85 * row_count)

    dataset['train'] = Dataset.from_pandas(dataset_df[:train_row_count][['text']])
    dataset['valid'] = Dataset.from_pandas(dataset_df[train_row_count:][['text']])

    return dataset


def train_llm_with_dataset_df(dataset_df, lora_llm, tokenizer, num_train_epochs, save_model_dir):
    """
    Train LLM (Large Language Model) using given Dataset DataFrame.
    Create Date: 2026.02.20

    :param dataset_df:       Dataset DataFrame
    :param lora_llm:         LLM to fine-tune
    :param tokenizer:        Tokenizer of LLM to fine-tune
    :param num_train_epochs: Train Epoch count
    :param save_model_dir:   Directory to save fine-tuned LLM
    """

    dataset = generate_llm_trainable_dataset(dataset_df)

    # create data collator
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

    :param dataset_df: Dataset DataFrame
    :param lora_llm:   LLM to fine-tune
    :param tokenizer:  Tokenizer of LLM to fine-tune
    """

    dataset_df['text'] = dataset_df.apply(
        lambda x: f"{x['user_input']} {ANSWER_PREFIX} {ANSWER_START_MARK} {x['tool_call_output']}{tokenizer.eos_token}",
        axis=1
    )

    train_llm_with_dataset_df(dataset_df,
                              lora_llm,
                              tokenizer,
                              num_train_epochs=5,
                              save_model_dir='execute_tool_call_llm')


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
    lora_llm, tokenizer = get_llm(LLM_PATH)
    dataset_df = pd.read_csv('../toolcall_training_data.csv')

    fine_tune_execute_tool_call_llm(dataset_df, lora_llm, tokenizer)
    fine_tune_final_output_llm(dataset_df, lora_llm, tokenizer)

