import json
import random
from random import shuffle
from collections import defaultdict
from tqdm import tqdm
from os import makedirs
import pandas as pd
import torch
from torch import Tensor
import numpy as np

from transformers import RobertaTokenizerFast, RobertaModel, RobertaConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Paths to the Hugging Face repository files
#TODO: change paths
repository_path = "/content/math_pretrained_roberta/"
tokenizer_path = "/content/math_pretrained_roberta/token"
config_path = repository_path + "config.json"

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(
    tokenizer_path,
    add_special_tokens=True,
    max_length=512,
    pad_to_max_length=True,
    padding='max_length'
)

# Load model configuration
config = RobertaConfig.from_pretrained(config_path)

# Load model weights
model = RobertaModel.from_pretrained(repository_path, config=config).to(device)




def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


data_path = '../data_processing'
out_path = '../task1/training_files'
train_p = 0.9  # --> valid_p = 1 - train_p, no test set, because ARQMath provides test set

data = json.load(open(f'{data_path}/cleaned_with_links.json', encoding='utf-8'))
makedirs(out_path, exist_ok=True)





def encode_answer(answer):
    encoded_input = tokenizer(answer, padding=True, truncation=True, return_tensors="pt")
    encoded_input['input_ids'] = torch.cat((encoded_input['input_ids'], torch.tensor([[tokenizer.pad_token_id] * (512 - encoded_input['input_ids'].shape[1])])), dim=1)

    encoded_input['attention_mask'] = torch.cat((encoded_input['attention_mask'], torch.tensor([[tokenizer.pad_token_id] * (512 - encoded_input['attention_mask'].shape[1])])), dim=1)
    encoded_input['attention_mask'] = encoded_input['attention_mask'].type(torch.int64)
    return encoded_input


list_with_all_questions = []
# 1. Remove questions without answers
# 2. Group questions by tag
questions_with_answers = defaultdict(list)
for q in data:
    if 'answers' not in q:
        continue  # we only want questions with answers
    list_with_all_questions.append(q)
    for tag in q['tags']:
        questions_with_answers[tag].append(q)

# 3. Check number of questions for each tag
print('Questions with answers, sizes by tag:')
for tag in questions_with_answers:
    print(tag, len(questions_with_answers[tag]))

# 4. For each questions: get one correct answer (random out of all answers of this question) and one incorrect answer with from another question with the highest cosine similarity from a pool of n answers randomly chosen
correct_pairs = []
wrong_pairs = []
pool_size = 10
model.eval()

for d in tqdm(data):
    if 'answers' in d:
        correct_answer = random.choice(d['answers'])
        correct_pairs.append((d['title'] + ' ' + d['question'] , correct_answer, '1')) # Label 1 for correct question-answer pairs
    #choose pool_size random questions from list_with_all_questions
    pool = random.sample(list_with_all_questions, pool_size + 1)
    #remove d from pool
    pool.remove(d)
    #pool answers
    pool = [random.choice(q['answers'] )for q in pool]
    #select for each qestion in pool 1 answer and append encoded to a list
    pool_encoded = [ encode_answer(q) for q in pool]
    input_ids = []
    attention_masks = []
    for sample in pool_encoded:
        input_ids.append(sample["input_ids"])
        attention_masks.append(sample["attention_mask"])
    # Concatenate input tensors
    input_ids = torch.cat(input_ids, dim=0).to(torch.int64)

    attention_masks = torch.cat(attention_masks, dim=0).to(torch.int64)
    pool_embeddings = []

    with torch.no_grad():
        outputs = model(input_ids.to(device), attention_mask=attention_masks.to(device))
        out_put_correct = model(encode_answer(correct_answer)["input_ids"].to(device), attention_mask=encode_answer(correct_answer)["attention_mask"].to(device))
    pool_embeddings.extend(outputs.last_hidden_state[:,0,:].cpu().numpy())

    scores = cos_sim(out_put_correct.last_hidden_state[:,0,:], pool_embeddings)

    highest_score_index = torch.topk(scores[0], 1).indices[1].item()


    wrong_answer = pool[highest_score_index]
    wrong_pairs.append((d['title'] + ' ' + d['question'], wrong_answer, '0')) # Label 0 for correct question-answer pairs


# 5. Shuffle data and save splits to file
all_pairs = [*correct_pairs, *wrong_pairs]
shuffle(all_pairs)

no_all = len(all_pairs)
no_train = int(no_all * 0.9)
no_val = no_all - no_train


def build_split(split, data_pairs):
    df = pd.DataFrame(data_pairs, columns=['question', 'answer', 'label'])
    df.to_csv(f'{out_path}/arqmath_task1_{split}.csv', index_label='idx')

build_split('train', all_pairs[:no_train])
build_split('dev', all_pairs[no_train:])
build_split('test', [])

print('Done creating training data for task 1.')