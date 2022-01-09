# PYTHON OFR EXTRA FUNCTION NOT NEEDED

def MAP(query_set, vectorized_doc_set, rel_set):
    n = len(rel_set)
    
    Map = 0
    
    for queryId, rel_docs in rel_set.items():
        query = query_set[queryId]
        ap = AP(query, vectorized_doc_set, rel_docs)
        Map += ap
        
    return Map / n
    
    
def AP(vector_query, vectorized_doc_set, rel_docs):
    mj = len(rel_docs)
    
    scores = cosine_sim_score(vectorized_doc_set.values(), vector_query)
    ap = 0
    
    top_id = extract_top_docs_index(scores, mj)
    rel_docs = set(rel_docs)
    
    for k in range(1, mj+1):
        proposed_docs_id = set(top_id[0:k])
        n_match = len(proposed_docs_id.intersection(rel_docs))
        
        ap += n_match / k
        
    return ap / mj

def R_precision(vector_query, vectorized_doc_set, rel_docs):
    R = len(rel_docs)
    
    scores = cosine_sim_score(vectorized_doc_set.values(), vector_query)
    
    top_id = extract_top_docs_index(scores, R)
    
    rel_docs_id = set(rel_docs)
    retrieved_docs_id = set(top_id)
    
    return len(rel_docs_id.intersection(retrieved_docs_id)) / R

def Mean_R_precision(query_set, vectorized_doc_set, rel_set):
    n = len(rel_set)
    
    MRP = 0
    
    for queryId, rel_docs in rel_set.items():
        query = query_set[queryId]
        R_prec = R_precision(query, vectorized_doc_set, rel_docs)
        MRP += R_prec
        
    return MRP / n


def precision_recall_curve(query_idx, query_set, vector_docs, rel_set, v):
    qry = query_set[query_idx]
    
    if query_idx in rel_set:
        rel_docs = rel_set[query_idx]
        R = len(rel_docs)
    else:
        return 0
    
    vector_query = vectorize_query(query, v)
    scores = cosine_sim_score(vector_docs, vector_query)
    
    # order the index od docs rertieved
    N = len(vector_docs)
    proposed_docs = extract_top_docs_index(scores, N)
    
    recalls = []
    precisions = []
    
    curr_recall = 0
    curr_precision = 0
    
    k = 1
    while curr_recall < 1:
        retrieved_docs = proposed_docs[0:k]
        n_items_retrieved = k
        
        n_relevant_items_retrieved = len([i for i in retrieved_docs if i in rel_docs])
        
        curr_precision = n_relevant_items_retrieved / n_items_retrieved
        curr_recall = n_relevant_items_retrieved / R
        
        recalls.append(curr_recall)
        precisions.append(curr_precision)
        k += 1
        
    return recalls, precisions


def prec_rec_at_11_levels(query_idx, query_set, vector_docs, rel_set, v):
    qry = query_set[query_idx]
    
    if query_idx in rel_set:
        rel_docs = rel_set[query_idx]
        R = len(rel_docs)
    else:
        return 0
    
    vector_query = vectorize_query(query, v)
    scores = cosine_sim_score(vector_docs, vector_query)
    
    # order the index od docs rertieved
    N = len(vector_docs)
    proposed_docs = extract_top_docs_index(scores, N)
    
    recalls = []
    precisions = []
    
    curr_recall = 0
    curr_precision = 0
    
    k = 1
    while curr_recall < 1:
        retrieved_docs = proposed_docs[0:k]
        n_items_retrieved = k
        
        n_relevant_items_retrieved = len([i for i in retrieved_docs if i in rel_docs])
        
        curr_precision = n_relevant_items_retrieved / n_items_retrieved
        curr_recall = n_relevant_items_retrieved / R
        
        recalls.append(curr_recall)
        precisions.append(curr_precision)
        k += 1
    
    desired_recs = np.linspace(0,1,11)
    found_precs = []
    k = 0
    for r in desired_recs:
        while recalls[k] < r:
            k += 1
        
        found_precs.append(max(precisions[k:]))
    
    return desired_recs, found_precs