
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, TrainerCallback, \
                         TrainerState, TrainerControl
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import pandas as pd


LLM_PATH = 'midm_original_llm'
ANSWER_START_MARK = ' ### Answer:'
ANSWER_PREFIX = '(답변 시작)'
lora_llm = None
tokenizer = None


class ValidateCallback(TrainerCallback):

    def __init__(self, valid_dataset, max_length=128):
        super(ValidateCallback, self).__init__()
        self.valid_dataset = valid_dataset
        self.max_length = max_length

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        global lora_llm, tokenizer

        print('=== INFERENCE TEST ===')

        for valid_input in self.valid_dataset:
            valid_input_text = valid_input['text'].split(ANSWER_PREFIX)[0] + f' {ANSWER_PREFIX}'

            inputs = tokenizer(valid_input_text, return_tensors='pt').to(lora_llm.device)
            inputs = {'input_ids': inputs['input_ids'].to(lora_llm.device),
                      'attention_mask': inputs['attention_mask'].to(lora_llm.device)}

            outputs = lora_llm.generate(**inputs,
                                        max_length=self.max_length,
                                        do_sample=True,
                                        temperature=0.6,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id)
            llm_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            llm_answer = llm_answer[len(valid_input_text):]
            llm_answer = llm_answer.split(ANSWER_PREFIX)[-1]

            print(f'valid input: {valid_input_text}\nLLM answer: {llm_answer} (total tokens: {len(outputs[0])})')


def get_llm(llm_path: str):
    """
        Get Large Language Model (LLM) to Fine-Tune, for AI Agent.
        Create Date : 2026.02.20

        :param llm_path: Path of Large Language Model (LLM)
        :return: tuple of (LoRA LLM, tokenizer)
    """

    global lora_llm, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    if tokenizer.pad_token is None:
        print('pad token of tokenizer is None, so add pad token')
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    if tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = '<pad>'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='bfloat16'
    )

    llm = AutoModelForCausalLM.from_pretrained(llm_path,
                                               quantization_config=bnb_config,
                                               trust_remote_code=True)
    llm.resize_token_embeddings(len(tokenizer))
    llm.gradient_checkpointing_enable()
    llm = prepare_model_for_kbit_training(llm)

    # LoRA (Low-Rank Adaption) config
    lora_config = LoraConfig(
        r=16,                              # Rank of LoRA
        lora_alpha=16,
        lora_dropout=0.05,                 # Dropout for LoRA
        init_lora_weights="gaussian",      # LoRA weight initialization
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LLM"
    )
    lora_llm = get_peft_model(llm, lora_config)

    return lora_llm, tokenizer


def train_llm(llm, dataset, data_collator, save_model_dir='fine_tuned_llm', max_length=128):
    """
        Train LLM for AI Agent, with given dataset.
        Create Date : 2026.02.20

        :param llm:            Large Language Model (LLM) to train
        :param dataset:        Training + Valid Dataset of LLM,
                               in the form of {'train': (Train Dataset), 'valid': (Valid Dataset)}
        :param data_collator:  Data Collator for the Dataset
        :param save_model_dir: Directory to save fine-tuned LLM
        :param max_length:     Maximum number of LLM output tokens
    """

    training_args = TrainingArguments(
        learning_rate=0.0003,            # lower learning rate is recommended for Fine-Tuning
        output_dir='./output',
        overwrite_output_dir=True,
        num_train_epochs=5,              # temp
        per_device_train_batch_size=2,   # temp
        per_device_eval_batch_size=1,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=1,                 # temp
        bf16=True,                       # for GPU
        report_to=None                   # to prevent wandb API key request at start of Fine-Tuning
    )

    trainer = SFTTrainer(
        llm,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        processing_class=tokenizer,
        args=training_args,
        data_collator=data_collator,
        callbacks=[ValidateCallback(dataset['valid'], max_length=max_length)]
    )

    trainer.train()
    trainer.save_model(save_model_dir)


def generate_llm_trainable_dataset(dataset_df):
    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'train'][['text']])
    dataset['valid'] = Dataset.from_pandas(dataset_df[dataset_df['data_type'] == 'valid'][['text']])

    return dataset


if __name__ == '__main__':
    get_llm(LLM_PATH)

    # mock toy dataset for functionality test
    dataset_dict = {
        'data_type': ['train', 'train', 'train', 'train', 'train',
                      'train', 'train', 'train', 'valid', 'valid',
                      'valid', 'valid'],
        'input_data': ['안녕?', '잘 지내?', '뭐하고 지내?', '반가워', '안녕!',
                       '안녕 요즘 뭐해', '요즘 뭐해?', '오랜만이야', '반가워!', '안녕 반가워',
                       '안녕 뭐해?', '안녕 잘 지내?'],
        'output_data': ['너도 잘 지내?', '나야 잘 지내지', '데이터 학습하는 중이야', '나는 LLM이야', '반가워 나는 LLM이야',
                        '데이터 학습 중!', '데이터 학습하고 있어', '반가워!', '[valid]', '[valid]',
                        '[valid]', '[valid]']
    }
    dataset_df = pd.DataFrame(dataset_dict)
    dataset_df['text'] = dataset_df.apply(
        lambda x: f"{x['input_data']}{ANSWER_PREFIX} {ANSWER_START_MARK} {x['output_data']}{tokenizer.eos_token}",
        axis=1
    )
    dataset = generate_llm_trainable_dataset(dataset_df)

    # create data collator
    response_template = tokenizer.encode(ANSWER_START_MARK, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # train LLM
    train_llm(lora_llm, dataset, collator, max_length=32)
