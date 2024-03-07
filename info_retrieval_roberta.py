from transformers import RobertaTokenizer, RobertaModel
import torch
from scipy.spatial.distance import cosine

def get_roberta_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

def find_best_answers_roberta(question, prompts, model_name='roberta-base', top_n=3):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    question_embedding = get_roberta_embeddings(question, tokenizer, model)

    similarities = []
    for prompt in prompts:
        prompt_embedding = get_roberta_embeddings(prompt, tokenizer, model)
        similarity = 1 - cosine(question_embedding.numpy(), prompt_embedding.numpy())
        similarities.append((prompt, similarity))

    top_prompts = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    return top_prompts

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

top_prompts = find_best_answers_roberta(question, prompts)
print("Roberta: Top 3 Answers with Cosine Similarity Scores:")
for prompt, score in top_prompts:
    print(f"Score: {score:.4f}, Prompt: {prompt}")
