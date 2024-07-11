from sys import argv
model_dirs = [
    'tiiuae/falcon-7b-instruct',
    'tiiuae/falcon-7b',
    'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-7b-hf',
]

max_seq_length=4096
buffer = 128

# %%


from os import listdir, stat
from os.path import isfile

# %%


from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer


# %%


from evaluate import load

bertscore = load("bertscore")


# %%


import json

with open('data.json', encoding="utf-8") as f:
    data = json.load(f)

q_val = [13,17,2,26,34,40,46,52,58,66,7]
q_tst = [12,16,22,30,36,42,48,5,54,61,68]


# %%
inst_temp = "### Source:"
resp_temp = "### Target:"
inst_temp = "[INST]"
resp_temp = "[/INST]"

from torch import inference_mode

def generate(model, tokenizer, prompt):
    tokens_source = tokenizer(inst_temp, return_tensors='pt')['input_ids'][0][:-1]
    tokenized_prompt = tokenizer(inst_temp + prompt + resp_temp, return_tensors='pt')['input_ids'].cuda()
    #tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
    max_prompt_length = max_seq_length-buffer
    if tokenized_prompt.shape[1] > max_prompt_length:
        tokenized_prompt = tokenized_prompt[:,tokenized_prompt.shape[1]-max_prompt_length:]
        tokenized_prompt[0][:len(tokens_source)] = tokens_source

    with inference_mode():
        output = model.generate(tokenized_prompt,
        max_new_tokens=buffer,
        do_sample=True,
    #    top_k=1,
        num_beams=4,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)


# %%
def progressivePrompt(model, tokenizer, lines):
    output = []
    prompt = "Simplify:\n"
    for line in lines:
        prompt += "\nOriginal: %s\nSimple: "%line
        output.append(generate(model, tokenizer, prompt).strip())
        prompt += output[-1] + "\n"
    return output


# lines = ["Background: Our primary objective was to determine the rate of persistent Trichomonas infection among pregnant women posttreatment.",
# "The secondary objective was to determine if oral multidose metronidazole was associated with fewer cases of persistent Trichomonas compared with single-dose treatment.",
# "Methods: This is a retrospective cohort study of women diagnosed with genital Trichomonas vaginalis from 2008 to 2017.",
# "We calculated the rate of persistent Trichomonas by dividing the number of positive Trichomonas tests collected 21 days or longer posttreatment by the total number of women treated and retested.",
# "Bivariate analysis was performed to compare the rates of positive tests after single and multidose metronidazole.",
# "Multivariate logistic regression was used to evaluate factors associated with persistent infection.",
# "Results: Five hundred forty-two women with 565 pregnancies were diagnosed with Trichomonas infection.",
# "The majority of subjects were prescribed either single-dose (n = 352) or multidose metronidazole (n = 74).",
# "Posttreatment Trichomonas tests were collected 21 days or longer in 326 subjects and 44% (143) were positive.",
# "Rates of positive Trichomonas tests among women receiving single-dose and multidose regimens were similar (45% vs. 40%, P = 0.50).",
# "Women who had ≥1 pregnancy affected by Trichomonas infection were more likely to have a positive test posttreatment (adjusted odds ratio, 20.1; 95% confidence interval, 1.9-215.3).",
# "Obese women were less likely to have a positive test posttreatment (adjusted odds ratio, 0.3; 95% confidence interval, 0.1-0.9).",
# "Conclusions: Given high rates of positive Trichomonas tests and increased detection with nucleic acid amplification tests (NAATs), all pregnant women should be retested with NAATs approximately 3 weeks posttreatment.",
# "Further studies are needed to determine the most effective treatment of Trichomonas infection in pregnant women.",
# ]
# 
# progressivePrompt(model, tokenizer, lines)

# print(generate(model, tokenizer, 'Simplify:\n\nOriginal: Background: In antidepressant trials for pediatric patients with depression or anxiety disorders, the risk of suicidal events and other severe psychiatric adverse events such as aggression and agitation is increased with antidepressants relative to placebo.\nSimple:'))

# model = AutoModelForCausalLM.from_pretrained(
#     'model-Llama-2-13b-hf/checkpoint-27000',
#     cache_dir = cache_dir,
#     )
# model.cuda()
# 

# tokenizer = AutoTokenizer.from_pretrained('model-Llama-2-13b-hf/checkpoint-27000')#, max_length=max_seq_length, pad_to_multiple_of=2048, padding='longest', truncation='max_length', add_eos_token=True)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# 

# %%
import time
from torch.cuda import empty_cache
from gc import collect

def inference(ckpt,use8bit=False):
    out = 'model-%s/checkpoint-0'%ckpt.split('/')[-1]
    print(out)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        load_in_8bit=use8bit,
        #cache_dir = cache_dir,
        )
    if not use8bit:
        model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
            ckpt,
            max_length=max_seq_length,
            truncation='max_length',
            add_eos_token=False,
            )
    
    scores = []
    denom = 0
    
    o = open(out+'.tsv', 'w', encoding="utf-8")
    o2 = open(out+'.time', 'w')
    
    for i, q in data.items():
        if int(i) not in q_val:
            continue
        print(i)
        for pmid, abst in q.items():
            if 'question' in pmid:
                continue
            print(pmid)
            srcs = list(abst['abstract'].values())
            start = time.time()
            outputs = progressivePrompt(model, tokenizer, srcs)
            end = time.time()
            o2.write('\t'.join([pmid, str(end - start)])+'\n')
            for j, src in enumerate(srcs):
                refs = []
                for adpt in abst['adaptations'].values():
                    ref = list(adpt.values())[j]
                    if ref != '':
                        refs.append(ref)
                if refs:#any([ref != '' for ref in refs]):
                    #print(src, refs)
                    score = bertscore.compute(predictions=[outputs[j]], references=[refs], lang="en")['f1'][0]
                else:
                    score = 0
                o.write('\t'.join([i, pmid, str(j+1), str(score), outputs[j].replace('\n','\\n').replace('\t', '\\t')])+'\n')
            #break
        #break
    o.close()
    o2.close()
    del model, tokenizer
    collect()
    empty_cache()

for model_dir in model_dirs:
    use8bit = '13b' in model_dir
    print(model_dir)
    inference(model_dir,use8bit)
    #break

# def score(ckpt)
#     f = open(ckpt + '.val.tsv')
#     for line in f.readlines():
#         (q, pmid, l, text) = line.strip().split()
#         print(data[q][pmid])
#         for j, src in enumerate(srcs):
#                 refs = []
#                 for adpt in abst['adaptations'].values():
#                     refs.append(list(adpt.values())[j])
#                 #if any([ref != '' for ref in refs]):
#                     #print(src, refs)
#                 #    score = bertscore.compute(predictions=[outputs[j]], references=[refs], lang="en")['f1'][0]
#                 #else:
#                 #    score = 0
#                 o.write('\t'.join([i, pmid, str(j+1), str(score), outputs[j]])+'\n')
# 

# %%




