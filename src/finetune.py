
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from random import random

# -

from sys import argv
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer

model_name = argv[1]
use8bit = '13b' in model_name
steps = 50000 if '13b' in model_name else 75000
save_steps = 1000

multi = len(argv)>2

if multi:
    alpha = float(argv[2])
    print('MSCRT, alpha=%f'%alpha)
else:
    save_steps = 100
    steps = 900

print("Steps: %d"%steps)
if use8bit:
    print("8bit")

max_seq_length = 2048 if 'meditron' in model_name else 4096

access_token = # YOUR TOKEN HERE

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #cache_dir=cache_dir,
    load_in_8bit=use8bit,
    token=access_token,
#    load_in_4bit=True,
#    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    #cache_dir=cache_dir,
#    padding='max_length',
#    padding_side='left',
    add_eos_token=False,
    use_fast=False,
    max_length=max_seq_length
)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

inst_temp = "### Source:"
resp_temp = "### Target:"

collator = DataCollatorForCompletionOnlyLM(
    instruction_template=inst_temp,
    response_template=resp_temp,
    tokenizer=tokenizer,
    mlm=False
)

from datasets import Dataset
import json

with open('data.json', encoding="utf-8") as f:
    data = json.load(f)

q_val = [13,17,2,26,34,40,46,52,58,66,7]
q_tst = [12,16,22,30,36,42,48,5,54,61,68]

def generate(questions = []):
    for i, q in data.items():
        if questions and int(i) not in questions:
            continue
        for pmid, abst in q.items():
            if 'question' not in pmid:
                for _, adpt in abst['adaptations'].items():
                    prompt = inst_temp + 'Simplify:\n\nOriginal: '
                    for j, sent in adpt.items():
                        prompt += abst['abstract'][j] + '\nSimple: '
                        text = prompt + resp_temp + sent + tokenizer.eos_token
                        prompt += '%s\n\nOriginal: '%(abst['abstract'][j] if multi and random() < alpha else sent)
                        words = text.split()
                        if len(words) > int(max_seq_length/2):
                            text = ' '.join(words[-int(max_seq_length/2):])
                            text = inst_temp+text[len(inst_temp):]
                        yield {"text":text}
#list(generate())[0]
ds_trn = Dataset.from_generator(generate)
ds_val = Dataset.from_generator(lambda: generate(q_val))
ds_tst = Dataset.from_generator(lambda: generate(q_tst))
ds_trn = ds_trn.shuffle()
ds_val = ds_val.shuffle()
ds_tst = ds_tst.shuffle()
print(ds_val[0]['text'])
print(ds_val[1]['text'])
print(ds_tst[0]['text'])
print(ds_tst[1]['text'])

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

args = TrainingArguments(
    output_dir = "./model-" + model_name.split('/')[-1]+(('-multi-%s'%argv[2]) if multi else ''),
    per_device_train_batch_size = 1,
    save_steps = save_steps,
    logging_steps = 100,
    eval_steps=100,
    max_steps = steps,#epochs*len(ds_trn),
)

trainer = SFTTrainer(
    model,
    data_collator=collator,
    train_dataset=ds_trn,
    eval_dataset=ds_val,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length = max_seq_length,
    args=args,
)

trainer.train()#resume_from_checkpoint=True)

