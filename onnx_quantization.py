import torch
import numpy as np
import pandas as pd
import warnings
import functools
import operator
import random
import json
import os
import time
import pickle

from src.onnx_models_structure import *
from src.metrics import rouge
from src.ort_settings import get_onnx_runtime_sessions, get_model_paths
from src.onnx_model import OnnxT5
from src.quantization import quantize

from progress.bar import Bar
from transformers import AutoConfig
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from tqdm import tqdm
from pathlib import Path


# ignores all the warnings during conversion
warnings.filterwarnings("ignore")
Bar.check_tty = False

is_export_onnx = False
pretrained_model = Path("VietAI/vit5-base-vietnews-summarization")
onnx_path = Path("onnx_model")

pt_model = AutoModelForSeq2SeqLM.from_pretrained(str(pretrained_model))
model_config = AutoConfig.from_pretrained(str(pretrained_model))

pt_model.eval()

onnx_fp32_model_paths = get_model_paths(pretrained_model, onnx_path)

if is_export_onnx:
	(
		simplified_encoder,
		decoder_with_lm_head,
		decoder_with_lm_head_init,
	) = turn_model_into_encoder_decoder(pt_model)

	encoder_path, decoder_path, init_decoder_path = onnx_fp32_model_paths

	# Though these are dummy inputs, ORT optimizations do reference these values,
	# so it is worth using values as close to production as possible
	input_sequence_length = 1024
	dummy_batch_size = 1  # not configurable since only CPU
	enc_seq_length = input_sequence_length
	dec_seq_length = 1  # a decoder sequence length is always one because it's just the last generated token
	onnx_opset_version = 13
	input_ids = torch.ones(dummy_batch_size, enc_seq_length, dtype=torch.int64)
	attention_mask = torch.ones(dummy_batch_size, enc_seq_length, dtype=torch.int64)

	n_heads = model_config.num_heads
	d_kv = model_config.d_kv

	input_ids_dec = torch.ones(dummy_batch_size, dec_seq_length, dtype=torch.int64)
	attention_mask_dec = torch.ones(dummy_batch_size, dec_seq_length, dtype=torch.int64)
	enc_out = torch.ones(
		(dummy_batch_size, enc_seq_length, model_config.d_model), dtype=torch.float32
	)

	sa = torch.ones(
		(dummy_batch_size, n_heads, dec_seq_length, d_kv), dtype=torch.float32
	)  # 1, 8, 1, 64
	ca = torch.ones(
		(dummy_batch_size, n_heads, enc_seq_length, d_kv), dtype=torch.float32
	)  # 1, 8, variable, 64
	t5_block = (sa, sa, ca, ca)
	past_key_values = (t5_block,) * model_config.num_decoder_layers

	flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])

	decoder_all_inputs = tuple(
		[input_ids_dec, attention_mask_dec, enc_out] + flat_past_key_values
	)

	bar = Bar("Exporting to onnx...", max=3)

	# Exports to ONNX
	with torch.no_grad():
		decoder_inputs = [
			"input_ids",
			"encoder_attention_mask",
			"encoder_hidden_states",
		]

		pkv_input_names = ["pkv_{}".format(i) for i in range(len(flat_past_key_values))]

		decoder_input_names = decoder_inputs + pkv_input_names

		decoder_output_names = ["logits", "output_past_key_values"]

		dyn_axis_general = {0: "batch", 1: "sequence"}
		dyn_axis_pkv = {0: "batch", 2: "seq_length"}

		dyn_axis = {
			"input_ids": dyn_axis_general,
			"encoder_attention_mask": dyn_axis_general,
			"encoder_hidden_states": dyn_axis_general,
			"logits": dyn_axis_general,
			"output_past_key_values": dyn_axis_general,
		}

		dyn_pkv = {
			"pkv_{}".format(i): dyn_axis_pkv for i in range(len(flat_past_key_values))
		}

		dyn_axis_params = {**dyn_axis, **dyn_pkv}

		# decoder to utilize past key values:
		torch.onnx.export(
			decoder_with_lm_head,
			decoder_all_inputs,
			decoder_path.as_posix(),
			export_params=True,
			do_constant_folding=True,
			opset_version=onnx_opset_version,
			input_names=decoder_input_names,
			output_names=decoder_output_names,
			dynamic_axes=dyn_axis_params,
		)
		bar.next()

		torch.onnx.export(
			simplified_encoder,
			args=(input_ids, attention_mask),
			f=encoder_path.as_posix(),
			export_params=True,
			opset_version=onnx_opset_version,
			do_constant_folding=True,
			input_names=["input_ids", "attention_mask"],
			output_names=["hidden_states"],
			dynamic_axes={
				"input_ids": dyn_axis_general,
				"attention_mask": dyn_axis_general,
				"hidden_states": dyn_axis_general,
			},
		)
		bar.next()
		# initial decoder to produce past key values
		torch.onnx.export(
			decoder_with_lm_head_init,
			(input_ids_dec, attention_mask_dec, enc_out),
			init_decoder_path.as_posix(),
			export_params=True,
			opset_version=onnx_opset_version,
			input_names=[
				"input_ids",
				"encoder_attention_mask",
				"encoder_hidden_states",
			],
			output_names=["logits", "past_key_values"],
			dynamic_axes={
				# batch_size, seq_length = input_shape
				"input_ids": dyn_axis_general,
				"encoder_attention_mask": dyn_axis_general,
				"encoder_hidden_states": dyn_axis_general,
				"logits": dyn_axis_general,
				"past_key_values": dyn_axis_general,
			},
		)

		bar.next()
		bar.finish()


models_config = {}

for model_path in onnx_fp32_model_paths:
	model_name = model_path.as_posix()
	config_path = model_name.replace(".onnx", ".pkl")

	file_name = os.path.basename(model_name)
	file_name = os.path.splitext(file_name)[0]

	with open(config_path, 'rb') as pickle_load:
		quantize_config = pickle.load(pickle_load)

	nodes_to_quantize = quantize_config[5]  # test

	models_config[file_name] = {
		"model_path": model_path,
		"nodes_to_quantize": nodes_to_quantize,
	}

quantize(models_config, approach="dynamic")
