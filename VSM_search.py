import pickle
from fast_answer import fast_feedexp_ans, fast_feedmov_ans, fast_pseudoexp_ans, fast_pseudomov_ans, fast_std_ans
from pyfiglet import Figlet

#import warnings
#warnings.filterwarnings("ignore")

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

def ask_method_and_k():
    
    method = None
    print("Available methods:")
    print("  1: standard")
    print("  2: pseudo query expansion")
    print("  3: pseudo query drift")
    print("  4: feedback query expansion")
    print("  5: feedback query drift")
    while method not in [1,2,3,4,5]:
        method = int(input("Select method:\t"))
        
    k = int(input("How many results you want?\t"))
    print("\n")
        
    return method, k
    
def answer_query(query, k, method):
    if method == 1:
        ind = fast_std_ans(query=query, k=k, processed_doc_set=PROCESSED_DOC_SET, posting_list=POSTING_LIST, vocab_idf=VOCABULARY_IDF, remove_sw=REMOVE_SW, lemmatize=LEMMATIZE)
    elif method == 2:
        ind = fast_pseudoexp_ans(query=query, k=k, processed_doc_set=PROCESSED_DOC_SET, posting_list=POSTING_LIST,
                                 vocab_idf=VOCABULARY_IDF, vocab=VOCABULARY, remove_sw=REMOVE_SW, lemmatize=LEMMATIZE)
    elif method == 3:
        ind = fast_pseudomov_ans(query=query, k=k, processed_doc_set=PROCESSED_DOC_SET, posting_list=POSTING_LIST,
                                 vocab_idf=VOCABULARY_IDF, vocab=VOCABULARY, remove_sw=REMOVE_SW, lemmatize=LEMMATIZE)
    elif method == 4:
        ind = fast_feedexp_ans(query=query, k=k, doc_set=DOC_SET, processed_doc_set=PROCESSED_DOC_SET,
                               posting_list=POSTING_LIST, vocab_idf=VOCABULARY_IDF, vocab=VOCABULARY, remove_sw=REMOVE_SW, lemmatize=LEMMATIZE)
    elif method == 5:
        ind = fast_feedmov_ans(query=query, k=k, doc_set=DOC_SET, processed_doc_set=PROCESSED_DOC_SET, 
                               posting_list=POSTING_LIST, vocab_idf=VOCABULARY_IDF, vocab=VOCABULARY, remove_sw=REMOVE_SW, lemmatize=LEMMATIZE)
    else:
        raise Exception 
    
    print("\nRESULTS:")
    for i in ind:   
        print(" ", DOC_SET[i]['book_name'])
    print("\n")
    

if __name__ == "__main__":
    
    #doc_set, processed_doc_set, vocabulary, posting_list, vocabulary_idf = prepare()
        
    f = Figlet(font='slant')
    print(f.renderText('VSM Search'))
    
    METHOD, K = ask_method_and_k()
    
    print("-----------------------------------------\n")
    query = input("Insert query (exit to close):\t")
    
    while query != "exit":
        answer_query(query=query, k=K, method=METHOD)
        print("-----------------------------------------\n")
        query = input("Insert query (exit to close):\t")
        
    

    