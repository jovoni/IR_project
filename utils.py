import re
import numpy as np

def vectorize(docID, docs, postings, vocab, vocab_idf):
    vector = np.zeros(len(vocab))
    #vector = [0] * len(vocab)
    text = preprocess_doc(docs[docID])
    for t in tokenize(text):
        idx = vocab.index(t)
        vector[idx] = postings[t][docID] * vocab_idf[t]
        
    return vector / np.linalg.norm(vector)

def vectorize_query(q, vocab):
    vector = np.zeros(len(vocab))
    query = preprocess_query(q)
    for t in tokenize(query):
        idx = vocab.index(t)
        vector[idx] = 1
    return vector / np.linalg.norm(vector)
    
def CosineScore(q, docs, vocab_idf, postings):
    scores = [0 for d in docs]
    lenght = [len(d) for d in docs]
    for t in tokenize(q):
        if t in vocab_idf:
            w = vocab_idf[t]
            posting_list = postings[t]
            for doc, term_freq in posting_list.items():
                scores[doc] += w * term_freq
    scores = [scores[i] / lenght[i] for i in range(len(scores))]
    return scores 

to_lower = lambda x: x.lower()

def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub(r"http\S+", "", text)
    text=re.sub('[^a-z]',' ',text)
    return text

remove_extra_spaces = lambda x: re.sub(' +',' ',x)

def create_vocab_and_postings_list(docs):
    vocab = []
    postings = {}
    
    for i, d in enumerate(docs):
        if i % 100 == 0:
            print(f"processed {i} docs")
        processed_d = preprocess_doc(d)
        words = tokenize(processed_d)
        for word in words:
            if word not in vocab:
                vocab.append(word)
            wordId = vocab.index(word)
            wordId = word
            if wordId not in postings:
                postings[wordId] = {i:1}
            else:
                if i in postings[wordId]:
                    postings[wordId][i] += 1
                else:
                    postings[wordId][i] = 1
                    
    return vocab, postings

def tokenize(text):
    return list(text.split())

def preprocess_doc(d):
    return remove_extra_spaces(clean_text(to_lower(d)))

def preprocess_query(q):
    return remove_extra_spaces(clean_text(to_lower(q)))