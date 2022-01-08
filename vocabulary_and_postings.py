from preprocessing import *
import math
from tqdm import tqdm

def build_vocab_and_postings(processed_doc_set):
    """
    Accepts a set of preprocessed docs
    """
    #vocab = {}
    vocab = []
    postings = {}
    
    #token_idx = 0
    
    for docId, tokens in tqdm(processed_doc_set.items()):
        for tk in tokens:
            if tk not in vocab:
                 #vocab[tk] = token_idx
                 #token_idx += 1
                vocab.append(tk)
             #tokenId = vocab[tk]
           #if tokenId not in postings:
           #    postings[tokenId] = {docId:1}
           #else:
           #    if docId in postings[tokenId]:
           #        postings[tokenId][docId] += 1
           #    else:
           #        postings[tokenId][docId] = 1
            if tk not in postings:
                postings[tk] = {docId:1}
            else:
                if docId in postings[tk]:
                    postings[tk][docId] += 1
                else:
                    postings[tk][docId] = 1
    return vocab, postings       

def build_vocab_with_idf(vocab, postings, docs):
    vocab_idf = {}
    N = len(docs)
    
    for tokenID, token in tqdm(enumerate(vocab)):
        vocab_idf[token] = (tokenID, math.log(N / len(postings[token]), 10))
        
    return vocab_idf

#def add_idf_to_vocab(vocab, postings, docs):
#    new_vocab = {}
#    N = len(docs)
#    
#    for token, tokenId in vocab.items():
#        new_vocab[token] = (tokenId, math.log(N / len(postings[tokenId]), 10))
#    return new_vocab
        