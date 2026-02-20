
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

LLM_PATH = 'midm_original_llm'


def get_llm(llm_path: str):
    """
        Get Large Language Model (LLM) to Fine-Tune, for AI Agent.
        Create Date : 2026.02.20

        :param llm_path: Path of Large Language Model (LLM)
        :return:         Tuple of (Transformers LoRA LLM, Tokenizer for the LLM)
    """

    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='bfloat16'
    )

    llm = AutoModelForCausalLM.from_pretrained(llm_path,
                                               quantization_config=bnb_config,
                                               trust_remote_code=True)
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

    return (lora_llm, tokenizer)


if __name__ == '__main__':
    (lora_llm, tokenizer) = get_llm(LLM_PATH)

    print(f'LoRA LLM :\n{str(lora_llm)[:200]}')
    print(f'\n\ntokenizer :\n{str(tokenizer)[:200]}')


