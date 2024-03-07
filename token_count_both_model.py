import random
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, BertForSequenceClassification
import torch

finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-fls")
finbert_model = BertTokenizer.from_pretrained("yiyanghkust/finbert-fls")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# ======== BERT ========
bert_examples = []
count = 0

for token in bert_tokenizer.vocab:
    
    if any(i.isdigit() for i in token) and not ('unused' in token):
        count += 1
        if random.randint(0, 100) == 1:
            bert_examples.append(token)

bert_percent = float(count) / len(bert_tokenizer.vocab)

print('In BERT:    {:>5,} tokens ({:.2%}) include a digit.'.format(count, bert_percent))

# ======== FinBERT ========
finbert_examples = []
count = 0

for token in finbert_tokenizer.vocab:

    if any(i.isdigit() for i in token) and not ('unused' in token):
        count += 1 
        if random.randint(0, 100) == 1:
            finbert_examples.append(token)

finbert_percent = float(count) / len(finbert_tokenizer.vocab)

print('In FinBERT: {:>5,} tokens ({:.2%}) include a digit.'.format(count, finbert_percent))

print('')
print('Examples from BERT:', bert_examples)
print('Examples from FinBERT:', finbert_examples)
