import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, pipeline
from scipy.spatial.distance import cosine

finbert_model = BertModel.from_pretrained("ahmedrachid/FinancialBERT", output_hidden_states=True)
finbert_tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT")

bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True) 
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_word_indeces(tokenizer, text, word):
    '''
    Determines the index or indeces of the tokens corresponding to `word`
    within `text`. `word` can consist of multiple words, e.g., "cell biology".
    
    Determining the indeces is tricky because words can be broken into multiple
    tokens. I've solved this with a rather roundabout approach--I replace `word`
    with the correct number of `[MASK]` tokens, and then find these in the 
    tokenized result. 
    '''
    word_tokens = tokenizer.tokenize(word)
    masks_str = ' '.join(['[MASK]']*len(word_tokens))
    text_masked = text.replace(word, masks_str)
    input_ids = tokenizer.encode(text_masked)
    mask_token_indeces = np.where(np.array(input_ids) == tokenizer.mask_token_id)[0]

    return mask_token_indeces

####
def get_embedding(b_model, b_tokenizer, text, word=''):
    '''
    Uses the provided model and tokenizer to produce an embedding for the
    provided `text`, and a "contextualized" embedding for `word`, if provided.
    '''

    # If a word is provided, figure out which tokens correspond to it.
    if not word == '':
        word_indeces = get_word_indeces(b_tokenizer, text, word)
    encoded_dict = b_tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        return_tensors = 'pt',     # Return pytorch tensors.
                )

    input_ids = encoded_dict['input_ids']
    
    b_model.eval()
    bert_outputs = b_model(input_ids) 
    with torch.no_grad():

        outputs = b_model(input_ids)
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]

    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding = sentence_embedding.detach().numpy()

    if not word == '':
        word_embedding = torch.mean(token_vecs[word_indeces], dim=0)
        word_embedding = word_embedding.detach().numpy()
    
        return (sentence_embedding, word_embedding)
    else:
        return sentence_embedding
    



# Three sentences; query is more similar to A than B.
# text_query = "Investors are closely monitoring the stock market for signs of volatility and potential opportunities. Market participants are analyzing various factors, including interest rates, corporate earnings, and geopolitical events, to make informed investment decisions."
# text_A = "The recent economic downturn has raised concerns among investors about the stability of their portfolios. Many are seeking expert advice on asset allocation and risk management to mitigate potential losses."
# text_B = "Astronomers are observing a rare celestial event involving the collision of two distant galaxies. This event is providing valuable insights into the formation of galaxies in the early universe."

text_query = "The recent fluctuations in the stock market have had a significant impact on investor confidence, with many shareholders expressing concern over the volatility of their portfolios."
text_A = "Investor confidence is often shaken by market volatility, leading to rapid sell-offs and a bearish outlook among shareholders, particularly in response to unpredictable economic indicators and corporate earnings reports."
text_B = "Climate change has been affecting global weather patterns, leading to an increase in the frequency and severity of extreme weather events such as hurricanes, droughts, and floods."

emb_query = get_embedding(finbert_model, finbert_tokenizer, text_query)
emb_A = get_embedding(finbert_model, finbert_tokenizer, text_A)
emb_B = get_embedding(finbert_model, finbert_tokenizer, text_B)

sim_query_A = 1 - cosine(emb_query, emb_A)
sim_query_B = 1 - cosine(emb_query, emb_B)

print("'query' should be more similar to 'A' than to 'B'...\n")

print('FinBERT:')
print('  sim(query, A): {:.2f}'.format(sim_query_A))
print('  sim(query, B): {:.2f}'.format(sim_query_B))

emb_query = get_embedding(bert_model, bert_tokenizer, text_query)
emb_A = get_embedding(bert_model, bert_tokenizer, text_A)
emb_B = get_embedding(bert_model, bert_tokenizer, text_B)

sim_query_A = 1 - cosine(emb_query, emb_A)
sim_query_B = 1 - cosine(emb_query, emb_B)

print('')
print('BERT:')
print('  sim(query, A): {:.2f}'.format(sim_query_A))
print('  sim(query, B): {:.2f}'.format(sim_query_B))
