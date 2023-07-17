import argparse
import sys
import os
import json
import time

import torch
from transformers import AutoConfig, AutoTokenizer, LongformerModel
from transformers.models.longformer.configuration_longformer import LongformerOnnxConfig
from transformers.utils import cached_file, TensorType

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from examples.pytorch.longformer.model import FTLongformerEncoder


def parse_from_config(model_name):
    with open(cached_file(model_name, 'config.json'), 'r') as f:
        config = json.load(f)
    layer_num = config['num_hidden_layers']
    hidden_size = config['hidden_size']
    head_num = config['num_attention_heads']
    size_per_head = hidden_size // head_num
    intermediate_size = config['intermediate_size']
    # assume all local attn window are same size. TODO: Improve later
    local_attn_window_size = config['attention_window'][0]
    attn_scaler = 1.0 / (size_per_head ** 0.5)
    return (layer_num, hidden_size, head_num, size_per_head,
            intermediate_size, local_attn_window_size, attn_scaler)


def build_ft_longformer(model_name, layer_num, head_num, size_per_head,
                        intermediate_size, local_attn_window_size,
                        max_global_token_num, batch_size, seq_len,
                        attn_scaler, ft_longformer_lib, data_type):
    weights_file = cached_file(model_name, 'pytorch_model.bin')
    max_seq_len = 4096
    ft_encoder = FTLongformerEncoder(weights_file, layer_num, head_num, size_per_head,
                                     intermediate_size, local_attn_window_size,
                                     max_global_token_num, batch_size, seq_len,
                                     attn_scaler, ft_longformer_lib, data_type)
    ft_longformer = build_hf_longformer(model_name)
    if data_type == 'fp16':
        ft_longformer = ft_longformer.half()
    elif data_type == 'bf16':
        ft_longformer = ft_longformer.bfloat16()
    ft_longformer.cuda()
    ft_longformer.eval()
    ft_encoder.set_hf_plugin_mode(True)
    ft_longformer.encoder = ft_encoder
    return ft_longformer


def build_hf_longformer(model_name):
    hf_longformer = LongformerModel.from_pretrained(model_name)
    hf_longformer.cuda()
    hf_longformer.eval()
    return hf_longformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', required=True,
                        help='huggingface model name')
    parser.add_argument('-l', '--ft-longformer-lib', type=str, default=os.path.join(project_root, 'build', 'lib', 'libth_transformer.so'),
                        help='Path to fastertransformer longformer pytorch op lib')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('-s', '--sequence-length',
                        help='The sequence length to use. Defaults to 1024',
                        default=1024, type=int)
    parser.add_argument('-b', '--batch-size',
                        help='Batch size to use. Note, it just copy the single question and passage token to form a batch, just for performance test.',
                        default=1, type=int)
    parser.add_argument("-g", "--max-global-attention-num", default=128,
                        help="Max global attention token num from start of the sequence to the end.", type=int)
    parser.add_argument('-r', '--repeat-test-num',
                        help='If specified, will run inference several rounds, to test average performance.',
                        type=int,
                        default=None)
    args, _ = parser.parse_known_args()
    print("======== Arguments ========")
    print(args)

    # prepare model config and weights
    model_name = args.model_name
    ft_longformer_lib = args.ft_longformer_lib
    seq_len = args.sequence_length
    batch_size = args.batch_size
    repeat_num = args.repeat_test_num if args.repeat_test_num else 0
    max_global_token_num = args.max_global_attention_num

    (layer_num, hidden_size, head_num, size_per_head,
     intermediate_size, local_attn_window_size, attn_scaler) = parse_from_config(model_name)

    # fastertransformer longformer
    ft_longformer = build_ft_longformer(model_name, layer_num, head_num, size_per_head,
                                        intermediate_size, local_attn_window_size,
                                        max_global_token_num, batch_size, seq_len,
                                        attn_scaler, ft_longformer_lib, args.data_type)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    onnx_config = LongformerOnnxConfig(config)
    model_inputs = onnx_config.generate_dummy_inputs(
        tokenizer,
        framework=TensorType.PYTORCH,
        batch_size=batch_size,
        seq_length=seq_len,
    )
    model_inputs["global_attention_mask"] = torch.zeros_like(model_inputs["input_ids"])
    model_inputs["global_attention_mask"][:, ::(seq_len // max_global_token_num)] = 1
    for k, v in model_inputs.items():
        model_inputs[k] = v.to("cuda")

    # model_inputs = [model_inputs["input_ids"], model_inputs["attention_mask"], model_inputs["global_attention_mask"]]
    with torch.no_grad():
        # FT warmup
        for i in range(10):
            ft_longformer(**model_inputs)

        start = time.time()
        for i in range(repeat_num):
            ft_longformer(**model_inputs)
        stop = time.time()
        print("FasterTransformer Longformer encoder average latency {:.3f} second ({} iterations)".format((stop - start) / repeat_num, repeat_num))


if __name__ == '__main__':
    main()
