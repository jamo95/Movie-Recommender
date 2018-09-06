import recommender
import pandas as pd

# read in full dataset to pandas data frames
# code obtained from: http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/

user_columns = ['user_id']
users = pd.read_csv('ml-100k/u.user', sep='|', usecols=range(1), names=user_columns, encoding='latin-1')

movie_columns = ['movie_id']
movies = pd.read_csv('ml-100k/u.item', sep='|', usecols=range(1), names=movie_columns, encoding='latin-1')

rating_columns = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', usecols=range(3), names=rating_columns, encoding='latin-1')

# run each recommender on full dataset

recommender.memory_user_collab_filter(users, movies, ratings)

recommender.memory_item_collab_filter(users, movies, ratings)

recommender.model_mf_collab_filter(users, movies, ratings)

recommender.hybrid_filter(users, movies, ratings)

# Evaluations for 100k rating data set

print("User User RMSE is: ", recommender.evaluate(model="user_similarity", dataset="ml-100k"))

print("Movie Movie RMSE is: ", recommender.evaluate(model="movie_similarity", dataset="ml-100k"))

print("MF SGD RMSE is: ", recommender.evaluate(model="mf_sgd", dataset="ml-100k"))
