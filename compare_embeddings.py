import numpy as np
import torch
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

####
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

#### Retrieve the models and tokenizers for both BERT and FinBERT
    
# Retrieve FinBERT.
finbert_model = BertModel.from_pretrained("ProsusAI/finbert",
                                  output_hidden_states=True)
finbert_tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

finbert_model.eval()
# Retrieve generic BERT.
bert_model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True) 
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_model.eval()

#### Test out the function
text = "In the realm of financial services, there is a growing concern about the disparity in access to affordable financial products and services. Many individuals and communities face significant challenges in achieving financial stability due to the lack of affordability and access to basic banking services. Bridging this disparity and improving the Affordability of financial solutions is a critical goal for financial institutions and policymakers alike."
word = "disparity"

# Get the embedding for the sentence and the specified word.
(sen_emb, word_emb) = get_embedding(finbert_model, finbert_tokenizer, text, word)

print('Embedding sizes:')
print(sen_emb.shape)
print(word_emb.shape)

#### Calculate the cosine similarity of the two embeddings.
sim = 1 - cosine(sen_emb, word_emb)

print('Cosine similarity: {:.2f}'.format(sim))