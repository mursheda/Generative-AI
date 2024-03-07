from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define mean pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

models = [
    ('bert-base-uncased', 'BERT'),
    ('ProsusAI/finbert', 'FinBERT'),
    ('yseop/roberta-base-finance-hypernym-identification', 'RoBERTa')
]
def encode_sentences(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])

# sentences = [
#     "The stock market faced a significant downturn.",
#     "Government bonds are considered a safe investment.",
#     "Convertible bonds offer both the benefits of bonds and stocks.",
#     "The economic indicators suggest a recession.",
#     "Equities are traded in stock markets."]

# sentences = [
#     "Rising interest rates have a cooling effect on the housing market.",
#     "An increase in the federal funds rate often leads to higher mortgage rates.",
#     "Tech stocks rallied after the announcement of quarterly earnings.",
#     "Merger discussions between the two corporations have halted due to valuation disagreements.",
#     "The central bank's new policy aims to curb inflation by reducing money supply."
# ]

sentences = [
    "Merger discussions between two major tech firms have been initiated.",
    "A significant drop in oil prices was observed due to increased production.",
    "The central bank announces a rise in interest rates to combat inflation.",
    "Cryptocurrency values surge as investors seek alternatives to traditional markets.",
    "New regulations are introduced to enhance cybersecurity in online banking."
]


prompt_labels = [f'Prompt Input {i+1}' for i in range(len(sentences))]

output_folder = '9.Finacial_prompt_2_semanticSimilarity'
os.makedirs(output_folder, exist_ok=True)

for model_path, model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    def encode_sentences(model, tokenizer, sentences):
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        return mean_pooling(model_output, encoded_input['attention_mask'])
    
    sentence_embeddings = encode_sentences(model, tokenizer, sentences).numpy()

    cosine_similarities = np.dot(sentence_embeddings, sentence_embeddings.T)
    norms = np.sqrt(np.diag(cosine_similarities))
    cosine_similarities = cosine_similarities / norms[:, None]
    cosine_similarities = cosine_similarities / norms[None, :]

    max_similarity = np.max(cosine_similarities[np.triu_indices_from(cosine_similarities, k=1)])
    print(f'Highest similarity score for {model_name}: {max_similarity:.2f}')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_similarities, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=prompt_labels, yticklabels=prompt_labels)
    plt.title(f'Semantic Similarity Heatmap ({model_name})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()

    heatmap_path = os.path.join(output_folder, f'{model_name}_heatmap_2.png')
    plt.savefig(heatmap_path)
    plt.close()

    print(f'Heatmap saved to: {heatmap_path}')
