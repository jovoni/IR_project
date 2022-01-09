import numpy as np
from scoring import extract_top_docs_index
from preprocessing import preprocess_string 
from vectorization import vectorize_query
from scoring import cosine_sim_score

def vect_std_ans(query, k, vector_docs, v, remove_stop_words, lemmatize):
    query = preprocess_string(query, remove_stop_words, lemmatize)
    vq = vectorize_query(query, v)
    scores = cosine_sim_score(vector_docs.values(), vq)
    ind = extract_top_docs_index(scores, k)
    return ind

def propose_docs_with_vector_query(vq, k, vector_docs, v):
    scores = cosine_sim_score(vector_docs.values(), vq)
    ind = extract_top_docs_index(scores, k)
    return ind

def vect_pseudoexp_ans(query, k, vector_docs, postings, v, v_idf, processed_docs_set, remove_sw, lemmatize):
    
    TOP_K = 20
    TOP_TERMS = 30
    
    ind = vect_std_ans(query, TOP_K, vector_docs, v, remove_sw, lemmatize)
    
    top_docs = {i : processed_docs_set[i] for i in ind}
    
    terms_tf_idf = [0] * len(v)
    
    for docId, doc_text in top_docs.items():
        for tk in doc_text:
            new_tf = postings[tk][docId]
            tokenId, new_idf = v_idf[tk]
            new_tf_idf = new_tf * new_idf
            
            if terms_tf_idf[tokenId] < new_tf_idf:
                terms_tf_idf[tokenId] = new_tf_idf
            
    terms_tf_idf = np.asarray(terms_tf_idf)
    ind_best_tokens = terms_tf_idf.argsort()[::-1][:TOP_TERMS]
    
    tokens = [v[i] for i in ind_best_tokens]
    
    extra_terms = ' '.join(tokens)
    new_query = query + ' ' + extra_terms
    
    new_ind = vect_std_ans(new_query, k, vector_docs, v, remove_sw, lemmatize)
    
    return new_ind

def vect_pseudomov_ans(query, k, vector_docs, v, remove_stop_words, lemmatize, alpha = 1, beta = 0.75):
    
    TOP_K = 20
    # first proposal
    ind = vect_std_ans(query, TOP_K, vector_docs, v, remove_stop_words, lemmatize)
    
    # modify query 
    query = preprocess_string(query, remove_stop_words, lemmatize)
    vq = vectorize_query(query, v)
    centroid = np.zeros_like(vq)
    
    for i in ind:
        centroid += vector_docs[i]
        
    centroid = centroid / TOP_K
    
    new_query = alpha * vq + beta * centroid
    new_query = new_query / np.linalg.norm(new_query)
    
    new_ind = propose_docs_with_vector_query(new_query, k, vector_docs, v)
    
    return new_ind


def vect_feedmov_ans(query, k, doc_set, vector_docs, v, processed_docs_set, remove_stop_words, lemmatize, alpha = 1, beta = 0.75, gamma = 0.15):
    # proposal docs
    ind = vect_std_ans(query, k, vector_docs, v)
    top_docs = {i : processed_docs_set[i] for i in ind}
    
    # ask user for feedback:
    print("\nInsert feedback (1 = relevant, 0 = non relevant):\n")
    r_doc_idx = []
    nr_doc_idx = []
    for docId in top_docs.keys():
        doc_name = doc_set[docId]['book_name']
        is_relevant = int(input(f'  Is document {doc_name} relevant?'))
        print(is_relevant)
        if is_relevant == 1:
            r_doc_idx.append(docId)
        elif is_relevant == 0:
            nr_doc_idx.append(docId)
        else:
            raise Exception
            
    # modify query
    query = preprocess_string(query, remove_stop_words, lemmatize)
    vq = vectorize_query(query, v)  
    
    centroid_r = np.zeros_like(vq)
    centroid_nr = np.zeros_like(vq)
    
    for i in r_doc_idx:
        centroid_r += vector_docs[i]
    for i in nr_doc_idx:
        centroid_nr += vector_docs[i]
        
    if len(r_doc_idx) < 0:
        centroid_r = centroid_r / len(r_doc_idx)
    if len(nr_doc_idx) > 0:
        centroid_nr = centroid_nr / len(nr_doc_idx)
    
    new_query = alpha * vq + beta * centroid_r - gamma * centroid_nr
    new_query = new_query / np.linalg.norm(new_query)
    
    ind = propose_docs_with_vector_query(new_query, k, vector_docs, v)
    
    return ind

def vect_feedexp_ans(query, k, doc_set, processed_doc_set, vector_docs, posting_list, vocab_idf, vocab, remove_sw, lemmatize, TOP_TERMS=20):
    # propose docs
    ind = vect_std_ans(query, k, vector_docs, vocab, remove_sw, lemmatize)
    
    # ask user for feedback:
    print("\nInsert feedback (1 = relevant, 0 = non relevant):\n")
    r_doc_idx = []
    nr_doc_idx = []
    for docId in ind:
        doc_name = doc_set[docId]['book_name']
        is_relevant = int(input(f'  Is document {doc_name} relevant?'))
        print(is_relevant)
        if is_relevant == 1:
            r_doc_idx.append(docId)
        elif is_relevant == 0:
            nr_doc_idx.append(docId)
        else:
            raise Exception
    
    # extract top terms from these docs
    top_docs = {i: processed_doc_set[i] for i in r_doc_idx}
    
    terms_tf_idf = [0] * len(vocab_idf)
    
    for docId, doc_text in top_docs.items():
        for tk in doc_text:
            new_tf = posting_list[tk][docId]
            tokenId, new_idf = vocab_idf[tk]
            new_tf_idf = new_tf * new_idf
            
            if terms_tf_idf[tokenId] < new_idf:
                terms_tf_idf[tokenId] = new_tf_idf
                
    terms_tf_idf = np.asarray(terms_tf_idf)
    ind_best_tokens = terms_tf_idf.argsort()[::-1][:TOP_TERMS]
    
    tokens = [vocab[i] for i in ind_best_tokens]
    
    extra_terms = ' '.join(tokens)
    query_expanded = query + ' ' + extra_terms
    
    new_ind = vect_std_ans(query_expanded, k, vector_docs, vocab, remove_sw, lemmatize)
    
    return new_ind
    

def vect_smart_feedmov_ans(query, k, vector_docs, v, rel_docs, remove_sw, lemmatize, alpha = 1, beta = 0.75, gamma = 0.15):
    # propose docs
    ind = vect_std_ans(query, k, vector_docs, v, remove_sw, lemmatize)
    
    # the "smart" user will tell us which docs are relevant
    r_doc_idx = [idx for idx in ind if idx in rel_docs]
    nr_doc_idx = [idx for idx in ind if idx not in rel_docs]
    
    # modify query
    query = preprocess_string(query, remove_sw, lemmatize)
    vq = vectorize_query(query, v) 
    
    # compute centroid
    centroid_r = np.zeros_like(vq)
    centroid_nr = np.zeros_like(vq)
    
    for i in r_doc_idx:
        centroid_r += vector_docs[i]
    for i in nr_doc_idx:
        centroid_nr += vector_docs[i]
        
    if len(r_doc_idx) < 0:
        centroid_r = centroid_r / len(r_doc_idx)
    if len(nr_doc_idx) > 0:
        centroid_nr = centroid_nr / len(nr_doc_idx)
    
    new_query = alpha * vq + beta * centroid_r - gamma * centroid_nr
    new_query = new_query / np.linalg.norm(new_query)
    
    ind = propose_docs_with_vector_query(new_query, k, vector_docs, v)
    
    return ind


def vect_smart_feedexp_ans(query, k, processed_doc_set, vector_docs, posting_list, vocab_idf, vocab, rel_docs, remove_sw, lemmatize, TOP_TERMS=20):
    # propose docs
    ind = vect_std_ans(query, k, vector_docs, vocab, remove_sw, lemmatize)
    
    # the "smart" user will tell us which docs are relevant
    r_doc_idx = [idx for idx in ind if idx in rel_docs]
    
    # extract top terms from these docs
    top_docs = {i: processed_doc_set[i] for i in r_doc_idx}
    
    terms_tf_idf = [0] * len(vocab_idf)
    
    for docId, doc_text in top_docs.items():
        for tk in doc_text:
            new_tf = posting_list[tk][docId]
            tokenId, new_idf = vocab_idf[tk]
            new_tf_idf = new_tf * new_idf
            
            if terms_tf_idf[tokenId] < new_idf:
                terms_tf_idf[tokenId] = new_tf_idf
                
    terms_tf_idf = np.asarray(terms_tf_idf)
    ind_best_tokens = terms_tf_idf.argsort()[::-1][:TOP_TERMS]
    
    tokens = [vocab[i] for i in ind_best_tokens]
    
    extra_terms = ' '.join(tokens)
    query_expanded = query + ' ' + extra_terms
    
    new_ind = vect_std_ans(query_expanded, k, vector_docs, vocab, remove_sw, lemmatize)
    
    return new_ind
    
    