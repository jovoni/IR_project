from preprocessing import *
import math

def build_vocab_and_postings(processed_doc_set):
    """
    Accepts a set of preprocessed docs
    """
    #vocab = {}
    vocab = []
    postings = {}
    
    #token_idx = 0
    
    for docId, tokens in processed_doc_set.items():
        for tk in tokens:
            if tk not in vocab:
                vocab.append(tk)
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
    
    for tokenID, token in enumerate(vocab):
        vocab_idf[token] = (tokenID, math.log(N / len(postings[token]), 10))
        
    return vocab_idf
