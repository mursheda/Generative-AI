from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity

def get_cosine_similarity_scores(question, passages, model_name='bert-large-uncased', top_n=3):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    question_tokens = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        question_embedding = model(**question_tokens).last_hidden_state[:, 0, :]  # Use the [CLS] token embedding

    similarities = []

    for passage in passages:
        passage_tokens = tokenizer(passage, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            passage_embedding = model(**passage_tokens).last_hidden_state[:, 0, :]

        similarity = cosine_similarity(question_embedding, passage_embedding)
        similarities.append((passage, similarity.item()))

    top_passages_scores = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    return top_passages_scores

question = "How will the upcoming earnings reports affect the stock market?"
passages = [
    "Earnings reports exceeding expectations can lead to a bullish market response, driving stock prices up.",
    "If many companies report earnings below expectations, it could trigger a market sell-off.",
    "Upcoming earnings reports are likely to increase market volatility, with sharp movements in individual stock prices.",
    "Positive earnings reports from tech giants could significantly boost the tech sector and overall market sentiment.",
    "Investors may be cautiously optimistic, with selective stock gains based on strong earnings.",
    "A trend of declining earnings could signal economic slowdown, affecting investor confidence.",
    "Strong earnings reports could mitigate concerns over high valuation levels in some market segments.",
    "Market reaction may be muted if good earnings results are already priced in by investors.",
    "Unexpectedly high earnings from energy companies could lead to increased investments in renewable energy stocks.",
    "Financial sector earnings will be closely watched for signs of interest rate impact on profits.",
]

top_passages_scores = get_cosine_similarity_scores(question, passages)
print("BERT: Top 3 Passages with Cosine Similarity Scores:")
for passage, score in top_passages_scores:
    print(f"Score: {score:.4f}, Passage: {passage}")
