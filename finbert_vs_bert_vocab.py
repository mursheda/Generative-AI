from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import torch

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

text = "In the wake of economic uncertainty, investors are showing an increased interest in gold, with many seeking to diversify their portfolios to mitigate risk and capitalize on potential market gains."

bert_tokens = bert_tokenizer.tokenize(text)
finbert_tokens = finbert_tokenizer.tokenize(text)

while len(finbert_tokens) < len(bert_tokens):
    finbert_tokens.append("")

print('{:<12} {:<12}'.format("BERT", "FinBERT"))
print('{:<12} {:<12}'.format("----", "-------"))

for tup in zip(bert_tokens, finbert_tokens):
    print('{:<12} {:<12}'.format(tup[0], tup[1]))
