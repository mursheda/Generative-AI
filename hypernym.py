from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

model = SentenceTransformer('yseop/roberta-base-finance-hypernym-identification')

input_directory_path = '4.Financial_clean_to_wrangledCsv'
output_directory_path = '5.HypernymAnalysisCharts'

if not os.path.exists(output_directory_path):
    os.makedirs(output_directory_path)

hypernyms = [
    'Bonds', 'Forward', 'Funds', 'Future', 'MMIs', 'Option', 'Stocks', 'Swap',
    'Equity Index', 'Credit Index', 'Securities restrictions', 'Parametric schedules',
    'Debt pricing and yields', 'Credit Events', 'Stock Corporation',
    'Central Securities Depository', 'Regulatory Agency'
]

hypernym_embeddings = model.encode(hypernyms, convert_to_tensor=True)

for csv_file in glob.glob(os.path.join(input_directory_path, '*.csv')):
    df = pd.read_csv(csv_file)
    df['Hypernym'] = ''
    df['Score'] = ''
    df['Top 3 Hypernyms'] = ''
    hypernym_counts = {hypernym: 0 for hypernym in hypernyms}

    for i, row in df.iterrows():
        query_embedding = model.encode(row['Financial Text'], convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, hypernym_embeddings)[0]
        top_results = torch.topk(cos_scores, k=3)  # Get top 3 results

        try:
            top_hypernyms = [hypernyms[idx] for idx in top_results.indices if idx < len(hypernyms)]
        except IndexError as e:
            print(f"Invalid index encountered. Details: {e}")
            continue

        df.at[i, 'Top 3 Hypernyms'] = ', '.join(top_hypernyms)



    hypernym_sentences_counts = Counter([hypernym for sublist in df['Top 3 Hypernyms'] for hypernym in sublist.split(', ')])
    sorted_hypernym_sentences_counts = sorted(hypernym_sentences_counts.items(), key=lambda item: item[1], reverse=True)

    hypernyms, counts = zip(*sorted_hypernym_sentences_counts)

    colors = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(12, 7))
    wedges, texts, autotexts = ax.pie(counts, autopct=lambda pct: "{:.1f}%".format(pct) if pct > 0 else '', startangle=140, colors=colors)
    ax.set_title(f'Distribution of Sentences Across Hypernyms for {os.path.basename(csv_file)}')

    custom_legends = [plt.Line2D([0], [0], marker='o', color='w', label=hypernym,
                                  markerfacecolor=color, markersize=10) for hypernym, color in zip(hypernyms, colors)]
    ax.legend(handles=custom_legends, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.axis('equal')
    plt.tight_layout(pad=3.0) 

    chart_name = os.path.basename(csv_file).replace('.csv', '_hypernym_pie_chart.png')
    plt.savefig(os.path.join(output_directory_path, chart_name), bbox_inches='tight')  # Ensure the whole figure is saved
    plt.close()  

    print(f"Analysis and chart saved for {os.path.basename(csv_file)}")