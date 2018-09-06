import csv
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import os.path
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# memory-based filter using Amazon item-item collaborative filtering algorithm
def memory_item_collab_filter(users, movies, ratings, dataset="", version=""):

    # creating ratings matrix in pandas dataframe format
    if os.path.exists("results/ratings_matrix_" + version + "_" + dataset + ".csv"):
        ratings_matrix = pd.DataFrame.from_csv("results/ratings_matrix_" + version + "_" + dataset + ".csv", index_col=0)
    else:
        ratings_matrix = pd.DataFrame(0, index=users["user_id"].tolist(),columns=movies["movie_id"].tolist())
        for rating in ratings.iterrows():
            ratings_matrix.loc[rating[1]["user_id"],rating[1]["movie_id"]] = rating[1]["rating"]
        ratings_matrix.to_csv("results/ratings_matrix_" + version + "_" + dataset + ".csv")
    print(ratings_matrix)

    # calculate movie similarity matrix using cosine similarity
    if os.path.exists("results/movie_similarity_matrix_" + version + "_" + dataset + ".csv"):
        movie_similarity_matrix = pd.DataFrame.from_csv("results/movie_similarity_matrix_" + version + "_" + dataset + ".csv", index_col=0)
    else:
        movie_similarity_matrix = pd.DataFrame(index=ratings_matrix.columns.tolist(),columns=ratings_matrix.columns.tolist())
        for x in range(0, len(ratings_matrix.columns)):
            print(x)
            for y in range(0, len(ratings_matrix.columns)):
                vec_1 = ratings_matrix.iloc[:,x].tolist()
                vec_2 = ratings_matrix.iloc[:,y].tolist()
                if np.linalg.norm(vec_1) * np.linalg.norm(vec_2) == 0:
                    movie_similarity_matrix.iloc[x,y] = 0
                else:
                    movie_similarity_matrix.iloc[x,y] = 1-cosine(vec_1,vec_2)
        movie_similarity_matrix.to_csv("results/movie_similarity_matrix_" + version + "_" + dataset + ".csv")
    print(movie_similarity_matrix)

    # calculate predicted ratings using weighted sum of similarity values and existing ratings
    if os.path.exists("results/movie_similarity_predicted_ratings_" + version + "_" + dataset + ".csv"):
        predicted_ratings = pd.DataFrame.from_csv("results/movie_similarity_predicted_ratings_" + version + "_" + dataset + ".csv", index_col=0)
    else:
        max_rating = ratings_matrix.values.max()
        predicted_ratings = pd.DataFrame(index=users["user_id"].tolist(),columns=movies["movie_id"].tolist())
        for x in range(0, len(predicted_ratings.index)):
            for y in range(0, len(predicted_ratings.columns)):
                movie_similarity_sum = sum(movie_similarity_matrix.iloc[y,:])
                if movie_similarity_sum == 0:
                    predicted_rating = 0
                else:
                    predicted_rating = sum(ratings_matrix.iloc[x,:] * movie_similarity_matrix.iloc[y,:]) / movie_similarity_sum * max_rating
                predicted_ratings.iloc[x,y] = predicted_rating
        predicted_ratings.to_csv("results/movie_similarity_predicted_ratings_" + version + "_" + dataset + ".csv")
        predicted_ratings.columns = predicted_ratings.columns.astype(str)
    print(predicted_ratings)
    return predicted_ratings

# memory-based filter using user-user collaborative filtering algorithm
def memory_user_collab_filter(users, movies, ratings, dataset="", version =""):

    # creating ratings matrix in pandas dataframe format
    if os.path.exists("results/ratings_matrix_" + version + "_" + dataset + ".csv"):
        ratings_matrix = pd.DataFrame.from_csv("results/ratings_matrix_" + version + "_" + dataset + ".csv", index_col=0)
    else:
        ratings_matrix = pd.DataFrame(0, index=users["user_id"].tolist(),columns=movies["movie_id"].tolist())
        for rating in ratings.iterrows():
            ratings_matrix.loc[rating[1]["user_id"],rating[1]["movie_id"]] = rating[1]["rating"]
        ratings_matrix.to_csv("results/ratings_matrix_" + version + "_" + dataset + ".csv")
    print(ratings_matrix)

    # calculate user similarity matrix using cosine similarity
    if os.path.exists("results/user_similarity_matrix_" + version + "_" + dataset + ".csv"):
        user_similarity_matrix = pd.DataFrame.from_csv("results/user_similarity_matrix_" + version + "_" + dataset + ".csv", index_col=0)
    else:
        user_similarity_matrix = pd.DataFrame(index=ratings_matrix.index.tolist(),columns=ratings_matrix.index.tolist())
        for x in range(0, len(ratings_matrix.index)):
            for y in range(0, len(ratings_matrix.index)):
                user_similarity_matrix.iloc[x,y] = 1 - cosine(ratings_matrix.iloc[x,:].tolist(),ratings_matrix.iloc[y,:].tolist())
        user_similarity_matrix.to_csv("results/user_similarity_matrix_" + version + "_" + dataset + ".csv")
    print(user_similarity_matrix)

    # calculate predicted ratings using weighted sum of similarity values and existing ratings
    # calculate user similarity matrix using cosine similarity
    if os.path.exists("results/user_similarity_predicted_ratings_" + version + "_" + dataset + ".csv") :
        predicted_ratings = pd.DataFrame.from_csv("results/user_similarity_predicted_ratings_" + version + "_" + dataset + ".csv", index_col=0)
    else:
        max_rating = ratings_matrix.values.max()
        predicted_ratings = pd.DataFrame(index=users["user_id"].tolist(),columns=movies["movie_id"].tolist())
        for x in range(0, len(user_similarity_matrix.index)):
            for y in range(0, len(predicted_ratings.columns)):
                user_similarity_sum = sum(user_similarity_matrix.iloc[x,:])
                if user_similarity_sum == 0:
                    predicted_rating = 0
                else:
                    predicted_rating = sum(ratings_matrix.iloc[:,y] * user_similarity_matrix.iloc[:,x])/user_similarity_sum * max_rating
                predicted_ratings.iloc[x,y] = predicted_rating
                # print(predicted_rating)
        predicted_ratings.to_csv("results/user_similarity_predicted_ratings_" + version + "_" + dataset + ".csv")
        predicted_ratings.columns = predicted_ratings.columns.astype(str)
    print(predicted_ratings)
    return predicted_ratings

# model-based filter using Netflix matrix facotrization model with stochastic gradient descent
def model_mf_collab_filter(users, movies, ratings, dataset="", version ="", plot=False):

    # creating ratings matrix in pandas dataframe format
    if os.path.exists("results/mf_ratings_matrix_" + version + "_" + dataset + ".csv"):
        ratings_matrix = pd.DataFrame.from_csv("results/mf_ratings_matrix_" + version + "_" + dataset + ".csv", index_col=0)
    else:
        ratings_matrix = pd.DataFrame(index=users["user_id"].tolist(),columns=movies["movie_id"].tolist())
        for rating in ratings.iterrows():
            ratings_matrix.loc[rating[1]["user_id"],rating[1]["movie_id"]] = rating[1]["rating"]
        ratings_matrix.to_csv("results/mf_ratings_matrix_" + version + "_" + dataset + ".csv")
        ratings_matrix = pd.DataFrame.from_csv("results/mf_ratings_matrix_" + version + "_" + dataset + ".csv", index_col=0)
    print(ratings_matrix)

    # stochastic gradient descent to optimize latent feature values
    if os.path.exists("results/mf_sgd_predicted_ratings_" + version + "_" + dataset + ".csv"):
        predicted_ratings_dataframe = pd.DataFrame.from_csv("results/mf_sgd_predicted_ratings_" + version + "_" + dataset + ".csv", index_col=0)
    else:

        # setup hyper parameter values
        num_fac = 5
        step_size = 0.01
        iterations = 50
        user_features = np.random.normal(0, 0.1, (len(ratings_matrix.index), num_fac))
        movie_features = np.random.normal(0, 0.1, (len(ratings_matrix.columns), num_fac))
        error_sums = []
        for iteration in range(iterations):
            error_sum = 0
            print("Iteration ", iteration)
            for rating in ratings.iterrows():
                user = rating[1]["user_id"]
                movie = rating[1]["movie_id"]
                rating = rating[1]["rating"]

                user_idx = ratings_matrix.index.get_loc(user)
                movie_idx = ratings_matrix.columns.get_loc(str(movie))
                prediction = np.dot(user_features[user_idx],movie_features[movie_idx])
                error = rating - prediction

                # adjust parameters
                user_features[user_idx,:] += step_size * error * movie_features[movie_idx,:]
                movie_features[movie_idx,:] += step_size * error * user_features[user_idx,:]
                error_sum += error
            error_sums.append(error_sum)
            print("Error sum is : ", error_sum)
        if plot == True:
            plt.plot(range(len(error_sums)),error_sums)
            plt.suptitle('Prediction Error per Iteration')
            plt.show()
        predicted_ratings_matrix = np.matmul(user_features, movie_features.T)
        predicted_ratings_dataframe = pd.DataFrame(predicted_ratings_matrix, index=users["user_id"].tolist(),columns=movies["movie_id"].tolist())
        predicted_ratings_dataframe.to_csv("results/mf_sgd_predicted_ratings_" + version + "_" + dataset + ".csv")
    print(predicted_ratings_dataframe)

    return predicted_ratings_dataframe

# hybrid filter based on switching between memory to model based approach when sparsity is above a specified threshold
def hybrid_filter(users, movies, ratings, dataset="", version ="", sparsity_threshhold = 0.05):

    # calculate sparsity
    num_users = len(users["user_id"].tolist())
    num_movies = len(movies["movie_id"].tolist())
    total_ratings = float(num_users * num_movies)
    known_ratings = float(len(ratings["rating"]))

    # run memory or model approach based on sparsity
    if known_ratings / total_ratings < sparsity_threshhold:
        predicted_ratings = memory_item_collab_filter(users, movies, ratings, dataset=dataset, version=version)
    else:
        predicted_ratings = model_mf_collab_filter(users, movies, ratings, dataset=dataset, version=version)
    return predicted_ratings

# calculates rmse
def evaluate(model="mf_sgd", dataset="ml-100k"):

    total_rmse = 0
    # read in data to pandas data frames
    # code obtained from: http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/

    user_columns = ['user_id', 'age', 'gender', 'profession', 'post_code']
    users = pd.read_csv('ml-100k/u.user', sep='|', names=user_columns, encoding='latin-1')

    movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_link']
    movies = pd.read_csv('ml-100k/u.item', sep='|', usecols=range(5), names=movie_columns, encoding='latin-1')

    rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=rating_columns, encoding='latin-1')

    u1_base = pd.read_csv(dataset+'/u1.base', sep='\t', names=rating_columns, encoding='latin-1')
    u1_test = pd.read_csv(dataset+'/u1.test', sep='\t', names=rating_columns, encoding='latin-1')
    u2_base = pd.read_csv(dataset+'/u2.base', sep='\t', names=rating_columns, encoding='latin-1')
    u2_test = pd.read_csv(dataset+'/u2.test', sep='\t', names=rating_columns, encoding='latin-1')
    u3_base = pd.read_csv(dataset+'/u3.base', sep='\t', names=rating_columns, encoding='latin-1')
    u3_test = pd.read_csv(dataset+'/u3.test', sep='\t', names=rating_columns, encoding='latin-1')
    u4_base = pd.read_csv(dataset+'/u4.base', sep='\t', names=rating_columns, encoding='latin-1')
    u4_test = pd.read_csv(dataset+'/u4.test', sep='\t', names=rating_columns, encoding='latin-1')
    u5_base = pd.read_csv(dataset+'/u5.base', sep='\t', names=rating_columns, encoding='latin-1')
    u5_test = pd.read_csv(dataset+'/u5.test', sep='\t', names=rating_columns, encoding='latin-1')

    squared_prediction_errors = []
    rmse_values = []

    if model == "user_similarity":
        for x in range(5):
            version = str(x + 1)
            u_base_file = 'ml-100k/u' + version + '.base'
            u_test_file = 'ml-100k/u' + version + '.test'
            base = pd.read_csv(u_base_file, sep='\t', names=rating_columns, encoding='latin-1')
            test = pd.read_csv(u_test_file, sep='\t', names=rating_columns, encoding='latin-1')
            predicted_ratings = memory_user_collab_filter(users, movies, base, dataset=dataset, version=version)
            for rating in test.iterrows():
                predicted_rating = predicted_ratings.loc[rating[1]["user_id"],str(rating[1]["movie_id"])]
                squared_prediction_error = abs(predicted_rating - rating[1]["rating"]) ** 2
                squared_prediction_errors.append(squared_prediction_error)
            rmse_values.append(np.sqrt(np.mean(squared_prediction_errors)))

    elif model == "movie_similarity":
        for x in range(5):
            version = str(x + 1)
            u_base_file = 'ml-100k/u' + version + '.base'
            u_test_file = 'ml-100k/u' + version + '.test'
            base = pd.read_csv(u_base_file, sep='\t', names=rating_columns, encoding='latin-1')
            test = pd.read_csv(u_test_file, sep='\t', names=rating_columns, encoding='latin-1')
            predicted_ratings = memory_item_collab_filter(users, movies, base, dataset=dataset, version=version)
            for rating in test.iterrows():
                predicted_rating = predicted_ratings.loc[rating[1]["user_id"],str(rating[1]["movie_id"])]
                squared_prediction_error = abs(predicted_rating - rating[1]["rating"]) ** 2
                squared_prediction_errors.append(squared_prediction_error)
            rmse_values.append(np.sqrt(np.mean(squared_prediction_errors)))

    elif model == "mf_sgd":
        for x in range(5):
            version = str(x + 1)
            u_base_file = 'ml-100k/u' + version + '.base'
            u_test_file = 'ml-100k/u' + version + '.test'
            base = pd.read_csv(u_base_file, sep='\t', names=rating_columns, encoding='latin-1')
            test = pd.read_csv(u_test_file, sep='\t', names=rating_columns, encoding='latin-1')
            predicted_ratings = model_mf_collab_filter(users, movies, base, dataset=dataset, version=version)
            for rating in test.iterrows():
                predicted_rating = predicted_ratings.loc[rating[1]["user_id"],rating[1]["movie_id"]]
                squared_prediction_error = abs(predicted_rating - rating[1]["rating"]) ** 2
                squared_prediction_errors.append(squared_prediction_error)
            rmse_values.append(np.sqrt(np.mean(squared_prediction_errors)))

    total_rmse = np.mean(rmse_values)

    return total_rmse

# print("User User RMSE is: ", evaluate(model="user_similarity", dataset="ml-100k"))

# print("Movie Movie RMSE is: ", evaluate(model="movie_similarity", dataset="ml-100k"))

# print("MF SGD RMSE is: ", evaluate(model="mf_sgd", dataset="ml-100k"))
