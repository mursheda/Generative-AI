from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.spatial.distance import cosine
import numpy as np

model_name = "yiyanghkust/finbert-pretrain"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs[0]

def find_best_answers_with_scores(question, prompts, tokenizer, model, top_k=3):
    question_embedding = encode_text(question, tokenizer, model)
    prompt_embeddings = [encode_text(prompt, tokenizer, model) for prompt in prompts]

    similarities = [1 - cosine(question_embedding.squeeze().numpy(), prompt_emb.squeeze().numpy()) for prompt_emb in prompt_embeddings]

    top_k_indexes = np.argsort(similarities)[-top_k:][::-1]
    top_prompts_with_scores = [(prompts[i], similarities[i]) for i in top_k_indexes]

    return top_prompts_with_scores
question = "How will the upcoming earnings reports affect the stock market?"
prompts = [
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

best_answers_with_scores = find_best_answers_with_scores(question, prompts, tokenizer, model)
print("FinBERT: Best Answers with Scores:")
for answer, score in best_answers_with_scores:
    print(f"Score: {score:.2f}, Prompt: {answer}")
