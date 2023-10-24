# %%
import torch
import numpy as np
import pandas as pd
import warnings
import random
import time
import psutil

from src.onnx_models_structure import *
from src.metrics import rouge
from src.ort_settings import get_onnx_runtime_sessions, get_model_paths
from src.onnx_model import OnnxT5

from progress.bar import Bar
from transformers import AutoConfig
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from tqdm import tqdm
from pathlib import Path

# %%
# ignores all the warnings during conversion
warnings.filterwarnings("ignore")
Bar.check_tty = False


# %%
def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)
	torch.manual_seed(worker_seed)
	torch.cuda.manual_seed(worker_seed)
	torch.cuda.manual_seed_all(worker_seed)


g = torch.Generator()
g.manual_seed(0)

# %%
providers = [
	"CPUExecutionProvider",
]

if torch.cuda.is_available():
	providers.insert(0, "CUDAExecutionProvider")

data_path = Path("data/vietnews/test.tsv")
pretrained_model = Path("VietAI/vit5-base-vietnews-summarization")
onnx_path = Path("onnx_model")

is_fp32 = False
max_input_length = 1024
input_sequence_length = 256
batch_size = 32

tokenizer = AutoTokenizer.from_pretrained(str(pretrained_model))
pt_model = AutoModelForSeq2SeqLM.from_pretrained(str(pretrained_model))
model_config = AutoConfig.from_pretrained(str(pretrained_model))

pt_model.eval()

# %%

if is_fp32:
	onnx_fp32_model_paths = get_model_paths(pretrained_model, onnx_path)
	fp32_sessions = get_onnx_runtime_sessions(
		onnx_fp32_model_paths,
		provider=providers,
		n_threads=psutil.cpu_count(),
		default=False,
	)
	fp32_model = OnnxT5(model_config, fp32_sessions)

else:
	onnx_int8_model_paths = get_model_paths(pretrained_model, onnx_path, quantized=True)

	int8_sessions = get_onnx_runtime_sessions(
		onnx_int8_model_paths,
		provider=providers,
		n_threads=psutil.cpu_count(),
		default=False,
	)
	int8_model = OnnxT5(model_config, int8_sessions)

print("Loaded Model")


# %%
def preprocess_function(examples):
	model_inputs = tokenizer(
		examples["inputs"], max_length=max_input_length, truncation=True, padding=True
	)
 
	with tokenizer.as_target_tokenizer():
		labels = tokenizer(
			examples["labels"],
			max_length=input_sequence_length,
			truncation=True,
			padding=True,
		)
	model_inputs["labels"] = labels["input_ids"]
	model_inputs["input_ids"] = model_inputs["input_ids"]

	return model_inputs


# %%
input_lines = []
label_lines = []

limit_test_example = batch_size*3

with open(str(data_path)) as file:
	for i, line in enumerate(file):
		if i == limit_test_example:
			break

		line = line.strip().split("\t")
		input = line[0]
		input_lines.append(input + "</s>")
		label_lines.append(line[1])
	
	print("Loaded Dataset")

input_lines = input_lines
label_lines = label_lines
dict_obj = {"inputs": input_lines, "labels": label_lines}

dataset = Dataset.from_dict(dict_obj)
test_tokenized_datasets = dataset.map(
	preprocess_function, batched=True, remove_columns=["inputs"], num_proc=psutil.cpu_count()
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=pt_model, return_tensors="pt")

# %%
# df = pd.read_csv(data_path,sep='\t')
# df.head(3)

# %%
dataloader = torch.utils.data.DataLoader(
	test_tokenized_datasets,
	collate_fn=data_collator,
	batch_size=batch_size,
	worker_init_fn=seed_worker,
	generator=g,
)
warn_up = 2

fp32_predictions = []
fp32_references = []
fp32_runtimes = []

int8_predictions = []
int8_references = []
int8_runtimes = []

pbar = tqdm(total=len(dataloader), desc="Evaluate")

start_eval = time.time()

for batch in dataloader:
	if is_fp32:
		fp32_start = time.time()
		fp32_outputs = fp32_model.generate(
			input_ids=batch["input_ids"],
			max_length=input_sequence_length,
			attention_mask=batch["attention_mask"],
			early_stopping=True,
		)
		fp32_end = time.time()

		with tokenizer.as_target_tokenizer():
			fp32_outputs = tokenizer.batch_decode(
				fp32_outputs.tolist()
			)
	else:
		int8_start = time.time()
		int8_outputs = int8_model.generate(
			input_ids=batch["input_ids"],
			max_length=input_sequence_length,
			attention_mask=batch["attention_mask"],
			early_stopping=True,
		)
		int8_end = time.time()

		with tokenizer.as_target_tokenizer():
			int8_outputs = tokenizer.batch_decode(
				int8_outputs.tolist(),
				max_length=max_input_length
			)

	with tokenizer.as_target_tokenizer():
		labels = np.where(
			batch["labels"] != -100, batch["labels"], tokenizer.pad_token_id
		)
  
		actuals = tokenizer.batch_decode(
			labels,
			max_length=max_input_length
		)

	if i > warn_up:
		if is_fp32:
			fp32_runtimes.append(fp32_end - fp32_start)
		else:
			int8_runtimes.append(int8_end - int8_start)

	if is_fp32:
		fp32_predictions.extend(fp32_outputs)
		fp32_references.extend(actuals)
	else:
		int8_predictions.extend(int8_outputs)
		int8_references.extend(actuals)

	pbar.update(1)

pbar.close()

end_eval = time.time()

if is_fp32:
	fp32_rouges = rouge(targets=fp32_references, predictions=fp32_predictions)
	fp32_rougeL = round(fp32_rouges["rougeL"] * 100, 4)
	print(fp32_rougeL)
	print(round(np.mean(fp32_runtimes)/batch_size, 2))
else:
	int8_rouges = rouge(targets=int8_references, predictions=int8_predictions)
	int8_rougeL = round(int8_rouges["rougeL"] * 100, 4)

	print(int8_rougeL)
	print(round(np.mean(int8_runtimes)/batch_size, 2))

# %%

# rougeL_loss = round(fp32_rougeL / int8_rougeL, 4)
# latency_drop = round(np.mean(fp32_runtimes) / np.mean(int8_runtimes), 2)

# print(rougeL_loss, latency_drop)

print(f"Total eval time = {round(end_eval-start_eval, 2)} seconds")
