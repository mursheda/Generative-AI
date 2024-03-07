import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import torch

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-esg")
finbert_model = AutoModel.from_pretrained("yiyanghkust/finbert-esg")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

words = [
    'Stock market', 'Inflation rate', 'Wage Stagnation', 'Financial Disparity',
    'Foreign exchange rate', 'Liquidity', 'Capital gains', 'Credit Access',
    'Financial Literacy', 'Housing Affordability', 'Savings Rate Disparity',
    'Investment Discrepancy', 'Economic Mobility', 'Market Volatility Impact',
    'Pension Gap', 'Income Diversification', 'Wealth Concentration',
    'Inflationary Effects', 'Educational Funding Gap', 'Healthcare Affordability',
    'Carbon factor', 'Emissions', 'Energy efficiency and renewable energy',
    'Sustainable Transport', 'Injury frequency rate', 'Sustainable Food & Agriculture'
]

for word in words:
        
    print('\n\n', word, '\n')

    list_a = ["BERT:"]
    list_b = ["FinBERT:"]

    list_a.extend(bert_tokenizer.tokenize(word))
    list_b.extend(finbert_tokenizer.tokenize(word))

    while len(list_a) < len(list_b):
        list_a.append("")
    while len(list_b) < len(list_a):
        list_b.append("")

    df = pd.DataFrame([list_a, list_b])
    print(df)
