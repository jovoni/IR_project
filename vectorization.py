from preprocessing import *
import numpy as np

def vectorize_doc(docId, preprocessed_doc_set, postings, vocab_idf):
    vector = np.zeros(len(vocab_idf))
    text = preprocessed_doc_set[docId]
    
    for tk in text:
        tokenId, idf = vocab_idf[tk]
        tf = postings[tk][docId]
        vector[tokenId] = tf * idf
        
    return vector / np.linalg.norm(vector) 

def vectorize_query(processed_query, vocab):
    vector = np.zeros(len(vocab))
    
    for tk in processed_query:
        if tk in vocab:
            tokenId = vocab.index(tk)
            vector[tokenId] = 1
        
    return vector / np.linalg.norm(vector)