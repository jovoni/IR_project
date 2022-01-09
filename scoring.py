import numpy as np
from vectorization import vector_to_tokens

def cosine_sim_score(docs_vectors, query_vectors):
    scores = [np.dot(d, query_vectors) for d in docs_vectors]
    return scores

def fast_cosine_score(processed_doc_set, processed_query, posting_list, vocab_idf):
    scores = {docId: 0 for docId in processed_doc_set.keys()}
    
    for t in processed_query:
        if t in vocab_idf:
            _, idf = vocab_idf[t]
            p_list = posting_list[t]
            for docId, tf in p_list.items():
                scores[docId] += tf * idf
        
    return scores

def weighted_query_cosine_score(processed_doc_set, vector_query, posting_list, vocab_idf, vocab):
    scores = {docId: 0 for docId in processed_doc_set.keys()}
    
    processed_query = vector_to_tokens(vector_query, vocab)
    
    for t in processed_query:
        if t in vocab_idf:
            tokenId, idf = vocab_idf[t]
            p_list = posting_list[t]
            q_weight = vector_query[tokenId]
            for docId, tf in p_list.items():
                scores[docId] += tf * idf * q_weight
                
    return scores

def extract_top_docs_index(scores, k):   
    scores = np.asarray(scores)
    ind = scores.argsort()[::-1][:k]
    return ind