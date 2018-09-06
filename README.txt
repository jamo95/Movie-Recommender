GOOGLE DRIVE LINK TO FILES (INCLUDING DATASET AND RESULTS DIRECTORY STRUCTURE):

https://drive.google.com/drive/folders/18MTh9TMjJ1wLrBlxEKUBco8XT2_TroBM?usp=sharing

HOW TO RUN:

Run main.py file to run models on whole datasets as well as cross validation tests on all models. Use Python not Python3 to run.

RESULTS:

Results are outputted to results directory. Predictions for each model are written in the format as follows:

- memory_item_collab_filter: movie_similarity_predicted_ratings_(dataset)_(version).csv

- memory_user_collab_filter: user_similarity_predicted_ratings_(dataset)_(version).csv

- model_mf_collab_filter: mf_sgd_predicted_ratings_(dataset)_(version).csv

Other files have been saved in results directory i.e. similarity and rating matrices calculated to use in calculation of predictions, as well as saved images of results

INFORMATION ON WHAT IS IMPLEMENTED (Functions with recommendation algorithms are):

- memory_item_collab_filter

- memory_user_collab_filter

- model_mf_collab_filter

- hybrid_filter

OTHERNOTES:

All recommendation functions take in user, movie and rating pandas data frames format based on movie lens dataset.

Datasets assumed to be in ml-100k directory. Though would also work for ml-1m and other movielens datasets.

Evaluate function runs cross validation tests on which model depending on model argument value.



