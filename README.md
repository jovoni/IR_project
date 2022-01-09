# Vector Space Model IR
Information Retrieval exam's project

The tool implemented should retrieve a list of books that suit the query given by the user.

## How to run VSM search

* run `python3 VSM_search.py`
* select method of search
* choose number of docs to retrieve
* follow instructions and obtain results
* insert ` quit ` if you want to close the program

## Info about methods

5 different methods have been implemented for the documents retrieval:

1. Standard : straightforward vector space model retrieval
2. Pseudo query expansion : method in which the query is expanded considering the top terms of the top retrieved docs, assuming the latter are relevant
3. Pseudo query drift : method in which the query is modified in the vector space by considering the vector represenattion of the top retrieved docs
3. Feedback query expansion : method in which the query is expanded considering the top terms in the relevant documents, given the feedback of the user
4. Feedback query drift : method in which the query is modified in the vector space by considering the vector representation of the relevant documents, given the feedback of the user

## Info about code

The repository contains the following files:

* `SVM_search.py`, the file to actually execute the search
* `dataset_evaluation.ipynb`, a notebook to evaluate the different methods on an annotated dataset
* `evaluation.py`, functions for the evaluation of the dataset
* `fast_answer.py`, functions for the search without the need of vectorizing all docs
* `vect_asnwer.py`, functions for the search with the need of vectorizing all docs
* `preprocessing.py`, functions to preprocess text
* `scoring.py`, function to score documents
* `vectorization.py`, functions to vectorize docs and queries
* `vocabulary_and_postings.py`, function to create vocabulary and posting list
* `data`, folder containing both the dataset for evaluation and the books dataset