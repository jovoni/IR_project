import re
import nltk

MIN_WORD_KENGTH = 2

STOPWORDS = nltk.corpus.stopwords.words('english')
STEMMER = nltk.stem.PorterStemmer()

def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub(r"http\S+", "", text)
    text=re.sub('[^a-z]',' ',text)
    #text=re.sub(' +',' ',text)
    return text

def tokenize(text):
    return list(text.split())

def preprocess_string(txt, remove_stop_words = True, lemmatize = True):
    """
    Return a preprocessed tokenize text.
    """
    txt = txt.lower()  
    txt_clean = clean_text(txt)
    tokens = tokenize(txt_clean)
    
    if remove_stop_words:
        tokens = [tk for tk in tokens if tk not in STOPWORDS]
    
    if lemmatize:
        tokens = [STEMMER.stem(tk) for tk in tokens]
        
    tokens = [tk for tk in tokens if len(tk) >= MIN_WORD_KENGTH]
    
    return tokens