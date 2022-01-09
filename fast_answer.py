
import numpy as np
from preprocessing import preprocess_string
from scoring import fast_cosine_score, weighted_query_cosine_score
from vectorization import vectorize_doc, vectorize_query


def fast_std_ans(query, k, processed_doc_set, posting_list, vocab_idf, remove_sw, lemmatize):
    """
    Output : inidices of the k highest scoring docs in the preprocessed doc set 
    with respect to the fast cosine score    
    """
    processed_qry = preprocess_string(query, remove_sw, lemmatize)
    scores = fast_cosine_score(processed_doc_set=processed_doc_set, processed_query=processed_qry,
                               posting_list=posting_list, vocab_idf=vocab_idf)
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    ind = list(sorted_scores.keys())[:k]
    return ind


def fast_pseudoexp_ans(query, k, processed_doc_set, posting_list, vocab_idf, vocab, remove_sw, lemmatize, TOP_K=20, TOP_TERMS=30):
    
    ind = fast_std_ans(query=query, k=TOP_K, processed_doc_set=processed_doc_set,
                            posting_list=posting_list, vocab_idf=vocab_idf, remove_sw=remove_sw, lemmatize=lemmatize)
    
    # extract top terms from these docs
    top_docs = {i: processed_doc_set[i] for i in ind}
    
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
    
    new_ind = fast_std_ans(query=query_expanded, k=k, processed_doc_set=processed_doc_set,
                                posting_list=posting_list, vocab_idf=vocab_idf, 
                                remove_sw=remove_sw, lemmatize=lemmatize)
    
    return new_ind


def fast_pseudomov_ans(query, k, processed_doc_set, posting_list, vocab_idf, vocab, remove_sw, lemmatize, TOP_K=20, ALPHA=1, BETA=0.75):
    # first proposal
    ind = fast_std_ans(query=query, k=TOP_K, processed_doc_set=processed_doc_set,
                            posting_list=posting_list, vocab_idf=vocab_idf, remove_sw=remove_sw, lemmatize=lemmatize)

    # move query
    query = preprocess_string(query, remove_sw, lemmatize)
    vq = vectorize_query(query, vocab)
    centroid = np.zeros_like(vq)
    
    for i in ind:
        centroid += vectorize_doc(docId=i, preprocessed_doc_set=processed_doc_set, 
                                  postings=posting_list, vocab_idf=vocab_idf)
        
    centroid = centroid / TOP_K
    
    new_query = ALPHA * vq + BETA * centroid
    
    scores = weighted_query_cosine_score(processed_doc_set=processed_doc_set, vector_query=new_query, 
                                          posting_list=posting_list, vocab_idf=vocab_idf, vocab=vocab)
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    new_ind = list(sorted_scores.keys())[:k]
    
    return new_ind


def fast_feedexp_ans(query, k, doc_set, processed_doc_set, posting_list, vocab_idf, vocab, remove_sw, lemmatize, TOP_TERMS=20):
    # first proposal
    ind = fast_std_ans(query=query, k=k, processed_doc_set=processed_doc_set,
                            posting_list=posting_list, vocab_idf=vocab_idf, remove_sw=remove_sw, lemmatize=lemmatize)
    
    print("\Insert feedback (1 = relevant, 0 = non relevant):\n")
    r_doc_idx = []
    nr_doc_idx = []
    for docId in ind:
        doc_name = doc_set[docId]['book_name']
        is_relevant = int(input(f'  Is document "{doc_name}" relevant?  '))
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
    
    new_ind = fast_std_ans(query=query_expanded, k=k, processed_doc_set=processed_doc_set,
                                posting_list=posting_list, vocab_idf=vocab_idf, 
                                remove_sw=remove_sw, lemmatize=lemmatize)
    
    return new_ind


def fast_feedmov_ans(query, k, doc_set, processed_doc_set, posting_list, vocab_idf, vocab, remove_sw, lemmatize,  alpha = 1, beta = 0.75, gamma = 0.15):
    # first proposal
    ind = fast_std_ans(query=query, k=k, processed_doc_set=processed_doc_set,
                            posting_list=posting_list, vocab_idf=vocab_idf, remove_sw=remove_sw, lemmatize=lemmatize)
    
    print("\nInsert feedback (1 = relevant, 0 = non relevant):\n")
    r_doc_idx = []
    nr_doc_idx = []
    for docId in ind:
        doc_name = doc_set[docId]['book_name']
        is_relevant = int(input(f'  Is document "{doc_name}" relevant? '))
        if is_relevant == 1:
            r_doc_idx.append(docId)
        elif is_relevant == 0:
            nr_doc_idx.append(docId)
        else:
            raise Exception
        
    # modify query
    query = preprocess_string(query, remove_sw, lemmatize)
    vq = vectorize_query(query, vocab)
    
    # compute centroid
    centroid_r = np.zeros_like(vq)
    centroid_nr = np.zeros_like(vq)
    
    for i in r_doc_idx:
        centroid_r += vectorize_doc(docId=i, preprocessed_doc_set=processed_doc_set, 
                                  postings=posting_list, vocab_idf=vocab_idf)
    for i in nr_doc_idx:
        centroid_nr += vectorize_doc(docId=i, preprocessed_doc_set=processed_doc_set, 
                                  postings=posting_list, vocab_idf=vocab_idf)
        
    if len(r_doc_idx) < 0:
        centroid_r = centroid_r / len(r_doc_idx)
    if len(nr_doc_idx) > 0:
        centroid_nr = centroid_nr / len(nr_doc_idx)
    
    new_query = alpha * vq + beta * centroid_r - gamma * centroid_nr
    
    scores = weighted_query_cosine_score(processed_doc_set=processed_doc_set, vector_query=new_query, 
                                          posting_list=posting_list, vocab_idf=vocab_idf, vocab=vocab)
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    new_ind = list(sorted_scores.keys())[:k]
    
    return new_ind
    


def fast_smart_feedexp_ans(query, k, processed_doc_set, posting_list, vocab_idf, vocab, rel_docs, remove_sw, lemmatize, TOP_TERMS=20):
    # first proposal
    ind = fast_std_ans(query=query, k=k, processed_doc_set=processed_doc_set,
                            posting_list=posting_list, vocab_idf=vocab_idf, remove_sw=remove_sw, lemmatize=lemmatize)

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
    
    new_ind = fast_std_ans(query=query_expanded, k=k, processed_doc_set=processed_doc_set,
                                posting_list=posting_list, vocab_idf=vocab_idf, 
                                remove_sw=remove_sw, lemmatize=lemmatize)
    
    return new_ind
    
    


def fast_smart_feedmov_ans(query, k, processed_doc_set, posting_list, vocab_idf, vocab, rel_docs, remove_sw, lemmatize,  alpha = 1, beta = 0.75, gamma = 0.15):
    # first proposal
    ind = fast_std_ans(query=query, k=k, processed_doc_set=processed_doc_set,
                            posting_list=posting_list, vocab_idf=vocab_idf, remove_sw=remove_sw, lemmatize=lemmatize)
    
    # the "smart" user will tell us which docs are relevant
    r_doc_idx = [idx for idx in ind if idx in rel_docs]
    nr_doc_idx = [idx for idx in ind if idx not in rel_docs]
    
    # modify query
    query = preprocess_string(query, remove_sw, lemmatize)
    vq = vectorize_query(query, vocab)
    
    # compute centroid
    centroid_r = np.zeros_like(vq)
    centroid_nr = np.zeros_like(vq)
    
    for i in r_doc_idx:
        centroid_r += vectorize_doc(docId=i, preprocessed_doc_set=processed_doc_set, 
                                  postings=posting_list, vocab_idf=vocab_idf)
    for i in nr_doc_idx:
        centroid_nr += vectorize_doc(docId=i, preprocessed_doc_set=processed_doc_set, 
                                  postings=posting_list, vocab_idf=vocab_idf)
        
    if len(r_doc_idx) < 0:
        centroid_r = centroid_r / len(r_doc_idx)
    if len(nr_doc_idx) > 0:
        centroid_nr = centroid_nr / len(nr_doc_idx)
    
    new_query = alpha * vq + beta * centroid_r - gamma * centroid_nr
    
    scores = weighted_query_cosine_score(processed_doc_set=processed_doc_set, vector_query=new_query, 
                                          posting_list=posting_list, vocab_idf=vocab_idf, vocab=vocab)
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    new_ind = list(sorted_scores.keys())[:k]
    
    return new_ind
    
    

    



    
    