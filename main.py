import pickle
from pyfiglet import Figlet
from preprocessing import preprocess_string
from scoring import fast_cosine_score

REMOVE_SW = True
LEMMATIZE = True

with open('data/books.pkl', 'rb') as f:
    DOC_SET = pickle.load(f)
with open('data/processed_books.pkl', 'rb') as f:
    PROCESSED_DOC_SET = pickle.load(f)
with open('data/vocabulary.pkl', 'rb') as f:
    VOCABULARY = pickle.load(f)
with open('data/posting_list.pkl', 'rb') as f:
    POSTING_LIST = pickle.load(f)    
with open('data/vocabulary_idf.pkl', 'rb') as f:
    VOCABULARY_IDF = pickle.load(f)
    
K = 3

def ask_method():
    
    method = None
    print("Select method. 1 = standard, 2 = pseudo_feedback, 3 = user_feedback")
    while method not in [1,2,3]:
        method = int(input(""))
        
    return method
    
def answer_query(query):
    processed_query = preprocess_string(query, REMOVE_SW, LEMMATIZE)
    scores = fast_cosine_score(PROCESSED_DOC_SET, processed_query, POSTING_LIST, VOCABULARY_IDF)
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    best_docs = list(sorted_scores.keys())[:K]
    for d in best_docs:
        print(DOC_SET[d]['book_name'])
    print("\n")

if __name__ == "__main__":
    
    #doc_set, processed_doc_set, vocabulary, posting_list, vocabulary_idf = prepare()
        
    f = Figlet(font='slant')
    print(f.renderText('VSM Search'))
    
    #method = ask_method()
    #print(method)
    
    query = input("Insert query: (exit to close)\t")
    
    while query != "exit":
        answer_query(query)
        query = input("Insert query: (exit to close)\t")
        
    

    