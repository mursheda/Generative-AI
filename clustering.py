import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

input_folder = '4.Financial_clean_to_wrangledCsv'
output_folder = '6.1.Combined_clustering_results'
plot_folder = '7.Financial_wranglesCsv_2_cluster_plots'

os.makedirs(output_folder, exist_ok=True)

models_to_compare = {
    'BERT': 'bert-base-uncased',
    'FinBERT': 'yiyanghkust/finbert-pretrain',
    'RoBERTa': 'yseop/roberta-base-finance-hypernym-identification'
}

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_sentences(sentences, tokenizer, model):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])

def extract_cluster_keywords(texts, n_keywords=5):
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    sums = X.sum(axis=0)
    data = []
    for col, term in enumerate(words):
        data.append((term, sums[0,col]))
    ranking = pd.DataFrame(data, columns=['term','rank'])
    ranking.sort_values('rank', ascending=False, inplace=True)
    return ranking['term'].head(n_keywords).tolist()

def visualize_clusters(embeddings_array, labels, model_name, csv_file):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_array)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette="Set1", s=50, alpha=0.9)
    plt.title(f'{model_name} Clusters for {csv_file}')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    model_plot_folder = os.path.join(plot_folder, model_name)
    os.makedirs(model_plot_folder, exist_ok=True)
    
    plot_file = os.path.join(model_plot_folder, f"{model_name}_plot_{csv_file.replace('.csv', '')}.png")
    plt.savefig(plot_file)
    plt.close()
    
def process_file(csv_file):
    df_original = pd.read_csv(os.path.join(input_folder, csv_file))
    if 'Financial Text' not in df_original.columns:
        print(f"Skipping {csv_file}: 'Financial Text' column not found.")
        return
    
    results_df = df_original[['Financial Text']].copy()

    for model_name, model_path in models_to_compare.items():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)

        sentences = df_original['Financial Text'].tolist()
        sentence_embeddings = encode_sentences(sentences, tokenizer, model)
        embeddings_array = sentence_embeddings.detach().numpy()

        # Clustering
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings_array)
        results_df[f'Cluster_{model_name}'] = kmeans.labels_
        cluster_themes = {i: extract_cluster_keywords([sentences[j] for j in range(len(sentences)) if kmeans.labels_[j] == i]) for i in range(n_clusters)}
        results_df[f'Theme_{model_name}'] = results_df[f'Cluster_{model_name}'].apply(lambda x: ', '.join(cluster_themes[x]))
        visualize_clusters(embeddings_array, kmeans.labels_, model_name, csv_file)

    combined_csv_file = os.path.join(output_folder, f"combined_{csv_file}")
    results_df.to_csv(combined_csv_file, index=False)
    print(f"Combined results saved to {combined_csv_file}")

for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        print(f"Processing {file_name}...")
        process_file(file_name)


