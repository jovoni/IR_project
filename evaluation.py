
from fast_answer import *
from vect_answer import *
from vocabulary_and_postings import *

def load_dataset(dataset_name):
    ### Processing DOCUMENTS
    doc_set = {}
    doc_id = ""
    doc_text = ""
    with open(f'data/{dataset_name}.ALL') as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
    for l in lines:
        if l.startswith(".I"):
            doc_id = int(l.split(" ")[1].strip())-1
        elif l.startswith(".X"):
            doc_set[doc_id] = doc_text.lstrip(" ")
            doc_id = ""
            doc_text = ""
        else:
            doc_text += l.strip()[3:] + " " # The first 3 characters of a line can be ignored.    

        
    ### Processing QUERIES
    with open(f'data/{dataset_name}.QRY') as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
        
    qry_set = {}
    qry_id = ""
    for l in lines:
        if l.startswith(".I"):
            qry_id = int(l.split(" ")[1].strip()) -1
        elif l.startswith(".W"):
            qry_set[qry_id] = l.strip()[3:]
            qry_id = ""

    ### Processing QRELS
    rel_set = {}
    with open(f'data/{dataset_name}.REL') as f:
        for l in f.readlines():
            qry_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]) -1
            doc_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1])-1
            if qry_id in rel_set:
                rel_set[qry_id].append(doc_id)
            else:
                rel_set[qry_id] = []
                rel_set[qry_id].append(doc_id)
                
    return doc_set, qry_set, rel_set


def compute_MAP(relevance_set, proposed_set):
    """
    Compute Mean Average Precision for the
    proposed set of relevant docs given
    the real set of relevant docs for the queries
    """
    N = len(relevance_set)
    MAP = 0
    
    for idx, relevant_docs in relevance_set.items():
        AP = 0
        mj = len(relevant_docs)
        docs_proposed = proposed_set[idx]
        for k in range(1, mj+1):
            docs_proposed = proposed_set[idx][:k]
            AP += len(set(relevant_docs).intersection(set(docs_proposed))) / k
        MAP += AP / mj
    return MAP / N


def compute_MK10(relevance_set, proposed_set):
    """
    Compure mean precision at K.
    If less than K relevant docs are present for
    a query it computes the precision at R where
    R is the number of relevant docs for the query
    """
    N = len(relevance_set)
    MK10 = 0
    
    for idx, relevant_docs in relevance_set.items():
        R = len(relevant_docs)
        k = 10 if R < 10 else R
        docs_proposed = proposed_set[idx][:k]
        precK = len(set(relevant_docs).intersection(set(docs_proposed))) / k
        MK10 += precK
    return MK10 / N


def compute_MRP(relevance_set, proposed_set):
    """
    Compure mean precision at R,
    where R is the number of relevant 
    docs for each query
    """
    N = len(relevance_set)
    MRP = 0
    
    for idx, relevant_docs in relevance_set.items():
        RP = 0
        R = len(relevant_docs)
        docs_proposed = proposed_set[idx][:R]
        RP += len(set(relevant_docs).intersection(set(docs_proposed))) / R
        MRP += RP
    return MRP / N

def print_eval(doc_set, qry_set, rel_set, remove_sw, lemmatize):
    print(f"Remove stopwords = {remove_sw}, lemmatize = {lemmatize}")
          
    ### Prepocess DOCS
    processed_docs_set = {}
    for docId, doc in doc_set.items():
        processed_docs_set[docId] = preprocess_string(doc, remove_stop_words=remove_sw, lemmatize=lemmatize)
        
    ### Preprocess QUERY
    processed_qry_set = {}
    for qryId, qry in qry_set.items():
        processed_qry_set[qryId] = preprocess_string(qry, remove_stop_words=remove_sw, lemmatize=lemmatize)
        
    # Create vocabulary list and postings
    v, p = build_vocab_and_postings(processed_docs_set)
    # Create vocabulary with tokenId and inverse document frequency (idf)
    v_idf = build_vocab_with_idf(v, p, processed_docs_set)
    print(f"Vocabulary contains {len(v)} tokens\n")
    
    # VECTORIZE doc set and qry set
    vector_docs = {docID : vectorize_doc(docID, processed_docs_set, p, v_idf) for docID in processed_docs_set.keys()}
    #vector_queries = {qId : vectorize_query(q, v) for qId, q in processed_qry_set.items()}
    
    # VECT STD ANS
    proposed_docs = {}
    for idx, relevant_docs in rel_set.items():
        q = qry_set[idx]
        prop_docs = vect_std_ans(q, len(relevant_docs), vector_docs, v, remove_sw, lemmatize)
        proposed_docs[idx] = prop_docs
        
    MAP0 = compute_MAP(rel_set, proposed_docs)
    MRP0 = compute_MRP(rel_set, proposed_docs)
    MK = compute_MK10(rel_set, proposed_docs)
    print(f'Standard vect | MAP = {MAP0:.2f} \t MRP = {MRP0:.2f} \t MK10 = {MK:.2f}')

    # FAST STD ANS
    proposed_docs = {}
    for idx, relevant_docs in rel_set.items():
        q = qry_set[idx]
        prop_ind = fast_std_ans(q, len(relevant_docs), processed_docs_set, p, v_idf, remove_sw, lemmatize)
        proposed_docs[idx] = prop_ind
        
    MAP0 = compute_MAP(rel_set, proposed_docs)
    MRP0 = compute_MRP(rel_set, proposed_docs)
    MK = compute_MK10(rel_set, proposed_docs)
    print(f'Standard fast | MAP = {MAP0:.2f} \t MRP = {MRP0:.2f} \t MK10 = {MK:.2f}')
    
    print("\n")
    
    # VECT PSEUDO EXP
    proposed_docs = {}
    for idx, relevant_docs in rel_set.items():
        q = qry_set[idx]
        prop_docs = vect_pseudoexp_ans(q, len(relevant_docs), vector_docs, p, v, v_idf, processed_docs_set, remove_sw, lemmatize)
        proposed_docs[idx] = prop_docs
        
    MAP0 = compute_MAP(rel_set, proposed_docs)
    MRP0 = compute_MRP(rel_set, proposed_docs)
    MK = compute_MK10(rel_set, proposed_docs)
    print(f'Pseu exp vect | MAP = {MAP0:.2f} \t MRP = {MRP0:.2f} \t MK10 = {MK:.2f}')
    
    
    # FAST PSEUDO EXP
    proposed_docs = {}
    for idx, relevant_docs in rel_set.items():
        q = qry_set[idx]
        prop_docs = fast_pseudoexp_ans(q, len(relevant_docs), processed_docs_set, p, v_idf, v, processed_docs_set, remove_sw, lemmatize)
        proposed_docs[idx] = prop_docs
        
    MAP0 = compute_MAP(rel_set, proposed_docs)
    MRP0 = compute_MRP(rel_set, proposed_docs)
    MK = compute_MK10(rel_set, proposed_docs)
    print(f'Pseu exp fast | MAP = {MAP0:.2f} \t MRP = {MRP0:.2f} \t MK10 = {MK:.2f}')
    
    print("\n")
    
    # VECT PSEUDO MOV
    proposed_docs = {}
    for idx, relevant_docs in rel_set.items():
        q = qry_set[idx]
        prop_docs = vect_pseudomov_ans(q, len(relevant_docs), vector_docs, v, remove_sw, lemmatize)
        proposed_docs[idx] = prop_docs
        
    MAP0 = compute_MAP(rel_set, proposed_docs)
    MRP0 = compute_MRP(rel_set, proposed_docs)
    MK = compute_MK10(rel_set, proposed_docs)
    print(f'Pseu mov vect | MAP = {MAP0:.2f} \t MRP = {MRP0:.2f} \t MK10 = {MK:.2f}')

    # FAST PSEUDO MOV
    proposed_docs = {}
    for idx, relevant_docs in rel_set.items():
        q = qry_set[idx]
        prop_docs = fast_pseudomov_ans(q, len(relevant_docs), processed_docs_set, p, v_idf, v, remove_sw, lemmatize)
        proposed_docs[idx] = prop_docs
        
    MAP0 = compute_MAP(rel_set, proposed_docs)
    MRP0 = compute_MRP(rel_set, proposed_docs)
    MK = compute_MK10(rel_set, proposed_docs)
    print(f'Pseu mov fast | MAP = {MAP0:.2f} \t MRP = {MRP0:.2f} \t MK10 = {MK:.2f}')
    
    print("\n")
    
    # VECT SMART FEED MOV
    proposed_docs = {}
    for idx, relevant_docs in rel_set.items():
        q = qry_set[idx]
        prop_docs = vect_smart_feedmov_ans(q, len(relevant_docs), vector_docs, v, relevant_docs, remove_sw, lemmatize)
        proposed_docs[idx] = prop_docs
        
    MAP0 = compute_MAP(rel_set, proposed_docs)
    MRP0 = compute_MRP(rel_set, proposed_docs)
    MK = compute_MK10(rel_set, proposed_docs)
    print(f'Feed mov vect | MAP = {MAP0:.2f} \t MRP = {MRP0:.2f} \t MK10 = {MK:.2f}')
    

    # FAST SMART FEED MOV
    proposed_docs = {}
    for idx, relevant_docs in rel_set.items():
        q = qry_set[idx]
        prop_docs = fast_smart_feedmov_ans(q, len(relevant_docs), processed_docs_set, p, v_idf, v, relevant_docs, remove_sw, lemmatize)
        proposed_docs[idx] = prop_docs
        
    MAP0 = compute_MAP(rel_set, proposed_docs)
    MRP0 = compute_MRP(rel_set, proposed_docs)
    MK = compute_MK10(rel_set, proposed_docs)
    print(f'Feed mov fast | MAP = {MAP0:.2f} \t MRP = {MRP0:.2f} \t MK10 = {MK:.2f}')
    
    print("\n")
    
    # VECT SMART FEED EXP
    proposed_docs = {}
    for idx, relevant_docs in rel_set.items():
        q = qry_set[idx]
        prop_docs = vect_smart_feedexp_ans(q, len(relevant_docs), processed_docs_set, vector_docs, p, v_idf, v, relevant_docs, remove_sw, lemmatize, TOP_TERMS=20)
        proposed_docs[idx] = prop_docs
        
    MAP0 = compute_MAP(rel_set, proposed_docs)
    MRP0 = compute_MRP(rel_set, proposed_docs)
    MK = compute_MK10(rel_set, proposed_docs)
    print(f'Feed exp vect | MAP = {MAP0:.2f} \t MRP = {MRP0:.2f} \t MK10 = {MK:.2f}')
    
    # FAST SMART FEED EXP FEEDBACK
    proposed_docs = {}
    for idx, relevant_docs in rel_set.items():
        q = qry_set[idx]
        prop_docs = fast_smart_feedexp_ans(q, len(relevant_docs), processed_docs_set, p, v_idf, v, relevant_docs, remove_sw, lemmatize, TOP_TERMS=20)
        proposed_docs[idx] = prop_docs
        
    MAP0 = compute_MAP(rel_set, proposed_docs)
    MRP0 = compute_MRP(rel_set, proposed_docs)
    MK = compute_MK10(rel_set, proposed_docs)
    print(f'Feed exp fast | MAP = {MAP0:.2f} \t MRP = {MRP0:.2f} \t MK10 = {MK:.2f}')