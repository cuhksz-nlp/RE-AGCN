import os

import torch
import torch.distributed as dist

from model import WEIGHTS_NAME, CONFIG_NAME, VOCAB_NAME

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
def is_main_process():
    return get_rank() == 0

def save_zen_model(save_zen_model_path, model, processor, tokenizer):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(save_zen_model_path, WEIGHTS_NAME)
    output_config_file = os.path.join(save_zen_model_path, CONFIG_NAME)
    output_dict_file = os.path.join(save_zen_model_path, "dict.bin")
    output_vocab_file = os.path.join(save_zen_model_path, VOCAB_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
        "labels_dict":processor.labels_dict,
        "types_dict":processor.types_dict
    }, output_dict_file)
    with open(output_config_file, "w", encoding='utf-8') as writer:
        writer.write(model_to_save.config.to_json_string())
    tokenizer.save(output_vocab_file)