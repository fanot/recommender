from __future__ import absolute_import
import pytz
from skimage.metrics import mean_squared_error

from model.SvdModel import *

import fire
import numpy as np
import pandas as pd
import os

from model.options import BaseOptions
from datetime import datetime
from sklearn.metrics import mean_squared_error
import surprise
from surprise import Reader, Dataset, SVD

from Levenshtein import distance as levenshtein
from sklearn.metrics.pairwise import cosine_similarity
import regex as re

import warnings

warnings.filterwarnings("ignore")


class BaseModel(object):
    """
    Recommender system class
    which connected with service
    and use BaseModels' heirs
    """

    def __init__(self, options: BaseOptions):
        # BaseOptions instance
        self.options = options

        # BaseModel's heir instance
        self.model = None

        # user dataframe matrix
        self.users_matrix = None

        # movies dataframe matrix
        self.items_matrix = None

        # ratings train dataframe
        self.ratings_train = None

        # ratings test dataframe
        self.ratings_test = None

        # user-item matrix
        self.user_item_matrix = None

        # array of mean rating for every user
        self.mean_user_rating = None

        # array of standard deviation of rating for every user
        self.std_user_rating = None

        # number of users
        self.n_users = None

        # number of movies
        self.n_items = None

        # have model been trained indicator
        self.trained = False

        # matrix of surprise based SVD predictions
        self.surprise_matrix = None

        # movie to movie similarity matrix
        self.items_similarity_matrix = None
        logging.info('instance successfully inited')

    def warmup(self, model_type: str = 'SVD'):
        """Method that rebuild model and reload its predictions from file

        Parameters
        ----------

        model_type : string, possible values: "SVD"
        """

        logging.info(f'started warmup method, model_type:{model_type}, model_data_path:{self.options.model_data_path}')

        # model load
        if model_type == 'SVD':
            self.model = SvdModel()
        else:
            logger.error(f'NameError: Invalid model type: {model_type}')
            raise NameError('tried to load invalid model type: {model_type}')

        # model's data load
        if self.__is_model_exists(self.options.model_data_path):
            self.model.load_data(self.options)
        else:
            logger.warning(f'model doesn\'t exist: {self.options.model_data_path}')

        # load user/movie data
        self.users_matrix, self.n_users = self.__load_users_data(self.options.users_data_path)
        self.items_matrix, self.n_items = self.__load_items_data(self.options.items_data_path)

        # proceeding loaded data
        self.users_matrix = self.__proceed_users(self.users_matrix)
        self.items_matrix = self.__proceed_items(self.items_matrix)

        # renew statistics as we renew model
        self.options.current_accuracy = 'Model haven\'t been evaluated yet'
        self.options.datetime_accuracy_test = 'Model haven\'t been evaluated yet'

    def train(self, train_data_path: str = None):
        """Method that fit default model


        Parameters
        ----------

        train_data_path : string, path to train dataset
        """

        logger.info(f'started train method, {train_data_path}')
        self.warmup()

        # if given dataset is None - use default dataset from options
        if train_data_path is None:
            train_data_path = self.options.train_data_path
            logger.info(f'given train_data_path is none, train_data_path changed to options.train_data_path: {self.options.train_data_path}')

        # load ratings
        self.ratings_train = self.__load_ratings(train_data_path)

        # create movie-user matrix based on given ratings
        self.user_item_matrix = self.__create_user_item_matrix(self.users_matrix, self.items_matrix, self.ratings_train)

        # normalize movie-user matrix
        self.user_item_matrix, self.mean_user_rating, self.std_user_rating = self.__normalize_matrix(
            self.user_item_matrix)

        # fit model
        self.model.fit(self.user_item_matrix, self.options.n_vectors, self.mean_user_rating,
                       self.std_user_rating)

        # save model
        self.model.save(self.options.model_name, self.options)
        self.trained = True
        print('trained')
        return

    def __is_model_exists(self, model_data_path: str):
        """Method that check if model exists on given path


        Parameters
        ----------

        model_data_path : string, path to model's data
        """

        return os.path.exists(model_data_path)

    def __create_user_item_matrix(self, users: pd.DataFrame, items: pd.DataFrame, ratings: pd.DataFrame):
        """Method that create user-movie matrix based on user, movie and ratings data


        Parameters
        ----------

        users: pd.DataFrame, users' dataframe
        items: pd.DataFrame, movies' dataframe
        ratings: pd.DataFrame, ratings' dataframe
        """

        logger.info(f'started __create_user_item_matrix method')

        # merge all data into one dataset
        user_item_rating_dataframe = self.__create_user_item_rating_dataframe(users, items, ratings)

        # create povit table(or matrix)
        matrix = user_item_rating_dataframe.pivot(index='user_id', columns='movie_id', values='rating')
        logger.info(f'__create_user_item_matrix method successfully executed')
        return matrix

    def __create_user_item_rating_dataframe(self, users: pd.DataFrame, items: pd.DataFrame, ratings: pd.DataFrame):
        """Method that merge user, movie and ratings data into one dataframe


        Parameters
        ----------

        users: pd.DataFrame, users' dataframe
        items: pd.DataFrame, movies' dataframe
        ratings: pd.DataFrame, ratings' dataframe
        """

        logger.info('started __create_user_item_rating_dataframe method')
        dataframe = pd.merge(ratings, items, on='movie_id', how='left').merge(users, on='user_id', how='left')
        logger.info(f'__create_user_item_rating_dataframe method successfully executed')
        return dataframe

    def __load_users_data(self, users_data: str):
        """Method that load users' data


        Parameters
        ----------

        users_data: string, path to users' data on disk
        """

        logger.info('started __load_users_data method')

        # read users data from disk
        users = pd.read_csv(users_data, names=['user_id', 'gender', 'age', 'occupation', 'zip-code'],
                            sep=self.options.data_loading_sep, engine=self.options.data_loading_engine,
                            encoding=self.options.encoding)

        # renew users number
        n_users = users['user_id'].nunique()

        logger.info('__load_users_data method successfully executed')
        return users, n_users

    def __load_items_data(self, items_data: str):
        """Method that load movies' data


        Parameters
        ----------

        items_data: string, path to movies' data on disk
        """

        # read movies data from disk
        items = pd.read_csv(items_data, names=['movie_id', 'title', 'genres'],
                            sep=self.options.data_loading_sep, engine=self.options.data_loading_engine,
                            encoding=self.options.encoding)

        # renew movies number
        n_items = items['movie_id'].nunique()

        return items, n_items

    def __load_ratings(self, ratings_data_path: str):
        """Method that load ratings' (test or train) data


        Parameters
        ----------

        ratings_data_path: string, path to ratings' data on disk
        """

        logger.info(f'started __load_ratings method, {ratings_data_path}')

        # read ratings data from disk
        ratings = pd.read_csv(ratings_data_path, names=['user_id', 'movie_id', 'rating', 'date'],
                              sep=self.options.data_loading_sep, engine=self.options.data_loading_engine,
                              encoding=self.options.encoding)

        logger.info(f'__load_ratings method successfully inited')
        return ratings

    def __proceed_items(self, items_matrix: pd.DataFrame):
        """Method that proceed loaded movies' dataframe


        Parameters
        ----------

        items_matrix: pd.DataFrame, movies' dataframe
        """

        logger.info('started __proceed_items method')

        # extract release_year year from movie title
        items_matrix['release_year'] = items_matrix['title'].str.extract(r'(?:\((\d{4})\))?\s*$', expand=False)

        logger.info('__proceed_items method successfully executed')
        return items_matrix

    def __proceed_users(self, users_matrix: pd.DataFrame):
        """Method that proceed loaded users' dataframe
        (now actually do nothing)

        Parameters
        ----------

        users_matrix: pd.DataFrame, users' dataframe
        """

        logger.info('started __proceed_users method')
        logger.info('__proceed_users method successfully executed')
        return users_matrix

    def __normalize_matrix(self, matrix: pd.DataFrame):
        """Method that normalize user-movie matrix(by users)

        Parameters
        ----------

        matrix: pd.DataFrame, user-movie matrix
        """

        logger.info('started __normalize_matrix method')

        # calculate user's mean rating
        mean_user_rating = np.nanmean(matrix.values, axis=1).reshape(-1, 1)

        # calculate user's standard deviation of rating
        std_user_rating = np.nanstd(matrix.values, axis=1).reshape(-1, 1)

        # normalize matrix
        matrix_normalized_values = (matrix.values - mean_user_rating) / std_user_rating

        # create matrix with normalized values and fill nans with 0
        matrix = pd.DataFrame(data=matrix_normalized_values, index=matrix.index, columns=matrix.columns).fillna(0)

        # fill nans with zero
        mean_user_rating[np.isnan(mean_user_rating)] = 0
        std_user_rating[np.isnan(std_user_rating)] = 0

        logger.info('__normalize_matrix method successfully executed')
        return matrix, mean_user_rating, std_user_rating

    def __normalize_row(self, row: pd.DataFrame):
        """Method that normalize users ratings' row

        Parameters
        ----------

        row: pd.DataFrame, row of matrix
        """

        logger.info('started __normalize_row method')

        # calculate user's mean rating
        mean_user_rating = np.nanmean(row.values, axis=1).reshape(-1, 1)

        # calculate user's standard deviation of rating
        std_user_rating = np.nanstd(row.values, axis=1).reshape(-1, 1)

        # if all ratings are the same and std equals 0
        # decrease mean and increase std for a little
        # otherwise model cannot calculate cosine similarity matrix
        if std_user_rating == 0:
            mean_user_rating -= 0.1
            std_user_rating += 0.1

        # normalize row values
        row_normalized_values = (row.values - mean_user_rating) / std_user_rating

        # create new row with normalized values and fill nans with 0
        row = pd.DataFrame(data=row_normalized_values, index=row.index, columns=row.columns).fillna(0)
        logger.info('__normalize_row method successfully executed')
        return row, mean_user_rating, std_user_rating

    def __get_movies_ids(self, predictions: pd.DataFrame):
        """Method that return all movies ids in given data

        Parameters
        ----------

        predictions: pd.DataFrame, matrix data
        """

        logger.info('started __get_movies_ids method')
        ids = predictions.columns.values
        logger.info('__get_movies_ids method successfully executed')
        return [int(x) for x in ids]

    def evaluate(self, test_data_path: str = None):
        """Method that return evaluate model

        Parameters
        ----------

        test_data_path: string, path to test dataset
        """

        logger.info('started evaluate method')

        # if model not trained - raise exception
        if self.trained == False:
            logger.error('evaluate method not executed as model not train')
            raise Exception('model not trained')

        # warmup model
        self.warmup()

        # if given dataset is None - use default from options
        if test_data_path == None:
            test_data_path = self.options.test_data_path

        # load test ratings
        self.ratings_test = self.__load_ratings(test_data_path)

        # create test dataset
        test_dataset = self.__create_user_item_rating_dataframe(self.users_matrix, self.items_matrix, self.ratings_test)

        # calculate rmse
        rmse = self.__calculate_rmse(test_dataset, self.model.data)

        print(f'RMSE: {rmse}')
        logger.info(f'evaluated with RMSE:{rmse}')

        # change model's info
        self.__set_evaluate_results(rmse)
        logger.info('evaluate method successfully executed')

        return rmse

    def __calculate_rmse(self, dataset: pd.DataFrame, preds: pd.DataFrame):
        """Method that calculate model RMSE

        Parameters
        ----------

        dataset: pd.DataFrame, test dataset
        preds: pd.DataFrame, models' predictions matrix
        """

        logger.info('started __calculate_rmse method')

        real_marks = []
        predictions = []

        # iterrate throught dataset and get predictions and real ratings
        for index, row in dataset.iterrows():
            user_id = row['user_id'] - 1
            movie_id = row['movie_id']
            rating = row['rating']
            # if movie_id in preds.columns:
            real_marks.append(rating)
            predictions.append(preds[movie_id][user_id])

        logger.info('__calculate_rmse method successfully executed')

        # calculate and return RMSE
        return mean_squared_error(real_marks, predictions, squared=False)

    def surprise_train(self, train_data_path: str = None):
        """Method that train model with sklearn-surprise based SVD


        Parameters
        ----------

        train_data_path : string, path to train dataset
        """

        logger.info('started surprise_train method')

        # if given dataset is None - use default dataset from options
        if train_data_path == None:
            train_data_path = self.options.train_data_path

        # load ratings
        self.ratings_train = self.__load_ratings(train_data_path)

        dataset = self.__surprise_get_dataset(self.ratings_train)

        # fit model
        self.__surprise_fit_model(dataset)

        self.trained = True
        print('trained')
        logger.info('surprise_train method successfully executed')
        return

    def __surprise_get_dataset(self, ratings: pd.DataFrame):
        """Method that create dataset for sklearn-surprise based model


        Parameters
        ----------

        ratings: pd.DataFrame, full user-movie-rating dataframe
        """

        return Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader=Reader(rating_scale=(1, 5)))

    def __surprise_fit_model(self, dataset: surprise.Dataset):
        """Method that create and fit sklearn-surprise based model


        Parameters
        ----------

         dataset: surprise.Dataset, train dataseе
        """

        logger.info('stared __surprise_fit_model method')

        # create model
        self.model = SVD(n_factors=50)

        # fit model
        self.model.fit(dataset.build_full_trainset())
        logger.info('__surprise_fit_model method successfully executed')

    def __surprise_make_predictions(self, dataset: surprise.Dataset):
        """Method that preding ratings from test dataset date


        Parameters
        ----------

         dataset: surprise.Dataset, test dataseе
        """


        logger.info('stared __surprise_make_predictions method')

        real_marks = []
        predictions = []

        # itterate throught dataset and get predictions and real ratings
        for row in dataset.build_full_trainset().build_testset():
            real_marks.append(row[2])
            predictions.append(self.model.predict(row[0], row[1]).est)

        logger.info('__surprise_make_predictions method successfully executed')
        return np.array(real_marks), np.array(predictions)

    def __surprise_calculate_rmse(self, real: np.matrix, pred: np.matrix):
        """Method that calculate RSME based on sklearn-suprise based model's predictions


        Parameters
        ----------

         real: np.matrix, real ratings
         pred: np.matrix, predicted ratings
        """

        return mean_squared_error(real, pred, squared=False)

    def surprise_evaluate(self, test_data_path: str = None):
        """Method that run sklearn-surprise based train and
        evaluate methods and return calculated RMSE

        Parameters
        ----------

        test_data_path : string, path to test dataset, model will train with dataset
        that stored in default train dataset path
        """

        logger.info('stared surprise_evaluate method')

        # if model haven't been trained yet - raise error
        if self.trained == False:
            logger.error('surprise_evaluate method not executed as model not train')
            raise Exception('surprise_evaluate: model not trained')

        # if given data is None - use default data from options
        if test_data_path == None:
            test_data_path = self.options.test_data_path

        # load test ratings
        self.ratings_test = self.__load_ratings(test_data_path)

        # create test dataset
        dataset = self.__surprise_get_dataset(self.ratings_test)

        # get real ratings and predicitons
        real_marks, predictions = self.__surprise_make_predictions(dataset)

        # calculate RMSE
        rmse = self.__surprise_calculate_rmse(real_marks, predictions)

        # renew model data
        self.__set_evaluate_results(rmse)
        logger.info(f'surprise: evaluated with RMSE:{rmse}')
        logger.info('surprise_evaluate method successfully executed')
        return rmse

    def __find_item_by_name(self, received_name: str):
        """Method to find most similar movie name to given
        basen on Levenshtein distance.

        Parameters
        ----------

        received_name: str, given movie name
        """

        # get index of movie with min Levenshtein distance from given name
        item_index = self.items_matrix['title'].apply(
            lambda title: levenshtein(re.sub(r' \([0-9]{4}\)', '', title.lower()), received_name.lower())).idxmin()

        # return this movie's title
        item_id = self.items_matrix.loc[item_index]['movie_id']
        return item_id

    def __calculate_items_similarity_matrix(self, items_matrix: pd.DataFrame):
        """Method that calculate movie to movie
        similarity matrix

        Parameters
        ----------

        items_matrix: pd.DataFrame, movie data
        """

        similarity_matrix = cosine_similarity(items_matrix, items_matrix)

        # zero diagonal because movie similarity to itself is equals to 1
        np.fill_diagonal(similarity_matrix, 0)

        similarity_df = pd.DataFrame(similarity_matrix, self.model.data.columns, self.model.data.columns)
        return similarity_df

    def __find_similar(self, movie_id: int, n: int = 5):
        """Method that find most similar {n} movies to given

        Parameters
        ----------

        movie_id: int, given movie ID
        n: int, amount of similar movies to return
        """

        # get most similar movie indexes without right order
        items_idxs = np.array(self.items_similarity_matrix[movie_id].sort_values(ascending=False)[:n].index.values,
                              dtype=int).tolist()

        # get most similar movies in right order
        items = self.__sort_items_by_ids(self.items_matrix, items_idxs)

        # get movie indexes in right order
        items_idxs = [int(x) for x in items_idxs]

        return items_idxs, items

    def get_similar_items(self, received_name: str = 'Bambi (1942)', amount: int = 5): ############
        """Method that find most similar {n} movies to given.
        Used to handle similar requests

        Parameters
        ----------

        received_name: str, given movie name
        amount: int, amount of similar movies to return
        """

        # find movie id
        item_id = self.__find_item_by_name(received_name)

        # calculate similarity matrix
        if self.items_similarity_matrix is None:
            self.items_similarity_matrix = self.__calculate_items_similarity_matrix(self.model.data.T)

        # return most similar movies
        items_idxs, items = self.__find_similar(item_id, amount)
        return [items_idxs, items]

    def predict(self, items_ratings: list, M: int = 10):
        """Method that predict top {m} movies for every user in test dataset.
        Used to handle predict requests.

        Parameters
        ----------

        items_ratings: list([movie_ids], [ratings]), double list of movie names and their ratings
        M: int, amount of similar movies
        """

        logger.info('started predict method')

        # if there are more than 2 arrays - raise error
        if len(items_ratings) != 2:
            logger.error('Wrong input: array dim must equals 2')
            raise ValueError('Wrong input: array dim is not equals 2')

        ratings = items_ratings[1]
        items_ids = items_ratings[0]

        # if there is movies ids array - convert ids into titles
        if not isinstance(items_ids[0], int):
            items_ids = [self.__find_item_by_name(x) for x in items_ids]

        data = [items_ids, ratings]

        # create new user with given ratings
        new_user_row = self.__init_new_row(data)

        # normalize row
        normalized_row, mean_user_rating, std_user_rating = self.__normalize_row(new_user_row)
        logger.info('normalized_row row in predict method')

        # calculate user similarity to other users
        weights = cosine_similarity(normalized_row.values * 1000, self.user_item_matrix * 1000).T
        logger.info('calcualte similarity matrix in predict')

        # calculate weighted marks
        weights = np.broadcast_arrays(weights, self.user_item_matrix)[0]
        output = np.average(self.user_item_matrix, axis=0, weights=weights)

        # denormalize mark
        marks = (output * std_user_rating + mean_user_rating)[0]

        logger.info('calculate marks in predict method')

        # get top movie ids
        best_movies_ids = np.argsort(-marks.reshape(1, -1), axis=1)[:, :M]
        best_movies_cols_ids = self.model.data.columns.values[best_movies_ids][0]

        # get top movie names
        films = self.__find_items_by_ids(best_movies_cols_ids)
        films.sort_values('movie_id', ascending=False)['title'].values.tolist()
        movies_names = self.__sort_items_by_ids(films, best_movies_cols_ids)
        logger.info('recieved best movies names in predict method')

        # get top movie ratings
        marks.sort()
        best_movies_marks = marks[::-1][:M]
        best_movies_marks[best_movies_marks > 5] = 5
        best_movies_marks[best_movies_marks < 0] = 0

        logger.info('predict method successfully executed')
        return [movies_names, best_movies_marks.tolist()]

    def predict_dataset(self, dataset_path: str, m: int = 5):
        """Method that predict top {m} movies for every user in test dataset.
        Used to handle CLI predict.

        Parameters
        ----------

        dataset_path: str, path to test dataset
        m: int, amount of similar movies
        """

        logger.info('started predict_dataset method')

        # if dataset is None - user default test dataset from options
        if dataset_path is None:
            dataset_path = self.options.test_data_path

        # load test ratings
        self.ratings_test = self.__load_ratings(self.options.test_data_path)

        # create user-movie matrix based on test ratings
        test_matrix = self.__create_user_item_matrix(self.users_matrix, self.items_matrix, self.ratings_test)

        # fill missed from train matrix values
        test_matrix = self.__fill_missed_values(test_matrix)

        # get top movies and their ratings
        top_items_names, top_items_ratings = self.__find_top_items_for_users(test_matrix, m)

        # generate dataframe
        result = self.__generate_dataframe(top_items_names, top_items_ratings, m)

        # save dataframe
        name = f'prediction_for_top_{m}_movies'
        result.to_csv(name)

        logger.info('predict_dataset method successfully executed')

        # return path to result dataframe
        return str(os.getcwd() + name)

    def __generate_dataframe(self, items_names: list, items_ratings: list, m: int = 5):
        """Method that genarate new dataframe based on
        given movie name, movie ratings and their amount
        Used to handle CLI predict.

        Parameters
        ----------

        items_names: list, array of movie names
        items_ratings: list, array of movie ratings
        m: int, amount of names and ratings
        """

        # get new columns
        columns = self.__generate_new_columns(m)

        data = []

        # generate rows
        for name, mark in zip(items_names, items_ratings):
            row_data = []
            for i in range(m):
                row_data.append(name[i])
                row_data.append(mark[i])
            data.append(row_data)

        # create and return new dataframe
        res_df = pd.DataFrame(data=data, columns=columns)
        res_df.index = res_df.index + 1

        logger.info('__generate_dataframe method successfully executed')
        return res_df

    def __find_top_items_for_users(self, test_matrix: pd.DataFrame, m: int = 5):
        """Method that find most similar {n} movies to every given user.
        Used to handle CLI predict.

        Parameters
        ----------

        test_matrix: pd.DataFrame, test dataset
        m: int, amount of similar movies for every user to return
        """

        # normalize test matrix
        normalized_matrix, mean_user_rating, std_user_rating = self.__normalize_matrix(test_matrix)

        # calculate user to user similarity
        user_to_user_similarity = cosine_similarity(normalized_matrix.values, self.user_item_matrix).T

        top_items_names = []
        top_items_ratings = []

        # find top movie for every test user
        for (i, user_data) in enumerate(zip(user_to_user_similarity, std_user_rating, mean_user_rating)):

            # find every movie estimated rating for every user
            # based on user to user similarity
            output = np.average(self.user_item_matrix, axis=0,
                                weights=np.broadcast_arrays(user_data[0].reshape(-1, 1), self.user_item_matrix)[0])

            # calculate estimated ratings
            output = output * user_data[1] + user_data[2]

            # find the best movies indexes
            best_movies_ids = np.argsort(-output.reshape(1, -1), axis=1)[:, :m]

            # find the best movies indexes in data
            best_movies_cols_ids = self.model.data.columns.values[best_movies_ids][0]

            # get top movie names
            items = self.__find_items_by_ids(best_movies_cols_ids)
            items.sort_values('movie_id', ascending=False)['title'].values.tolist()
            movies_names = self.__sort_items_by_ids(items, best_movies_cols_ids)

            # get top movies ratings
            best_movies_marks = np.sort(output)[::-1][:m]
            best_movies_marks[best_movies_marks > 5] = 5
            best_movies_marks[best_movies_marks < 0] = 0

            top_items_names.append(movies_names)
            top_items_ratings.append(best_movies_marks)

        logger.info('__find_top_items_for_users method successfully executed')
        return top_items_names, top_items_ratings

    def __generate_new_columns(self, m):
        """Method to generate columns pairs('movie_name_X', 'mark_X')
         for dataframe.
         Used to handle predict dataset


        Parameters
        ----------

        m: int, number of column pair
        """

        cols = []
        for i in range(1, m + 1):
            cols.append(f'movie_name_{i}')
            cols.append(f'mark_{i}')

        return cols

    def __fill_missed_values(self, test_matrix: pd.DataFrame):
        """Method that find missing users and movies
        in test user-movie matrix and add them to it
        with given ids

        Parameters
        ----------

        test_matrix: pd.DataFrame, test user-movie matrix
        """

        # missing movies
        train_columns = self.model.data.columns.values
        test_columns = test_matrix.columns.values
        movies_lack_in_test = set(train_columns) - set(test_columns)

        # missing users
        train_index = self.model.data.index.values
        test_index = test_matrix.index.values - 1
        users_lack_in_test = set(train_index) - set(test_index)

        # add new movies to matrix
        new_columns = np.append(test_columns, list(movies_lack_in_test))

        # add new users to matrix
        new_index = np.append(test_index, list(users_lack_in_test))

        # create new matrix with missing users filled with nans
        new_data = np.empty((len(users_lack_in_test), test_matrix.shape[1]))
        new_data[:] = np.nan
        matrix_filled_users = np.vstack([test_matrix.values, new_data])

        # create new matrix with missing users and movies filled with nans
        new_data = np.empty((matrix_filled_users.shape[0], len(movies_lack_in_test)))
        new_data[:] = np.nan
        new_data = np.hstack([matrix_filled_users, new_data])

        # sort new matrix
        filled_matrix = pd.DataFrame(data=new_data, columns=new_columns, index=new_index)
        filled_matrix = filled_matrix.sort_index()
        filled_matrix = filled_matrix.reindex(sorted(filled_matrix.columns), axis=1)

        logger.info('__fill_missed_values method successfully executed')
        return filled_matrix

    def __find_items_by_ids(self, ids: list):
        """Method that find movies from movie dataset
        with given ids

        Parameters
        ----------

        ids: list, list of movie ids
        """

        return self.items_matrix.loc[self.items_matrix['movie_id'].isin(ids), :]

    def __init_new_row(self, items_ratings: list):
        """Create new row from given data.
        Used to handle predict requests

        Parameters
        ----------

        items_ratings: list([], []), list of movie names and their ratings
        """

        items_ids = items_ratings[0]
        ratings = items_ratings[1]

        # create empty row
        values = np.empty((1, len(self.model.data.columns)))
        values.fill(np.nan)

        new_user_row = pd.DataFrame(data=values, columns=self.model.data.columns)

        # fill row with given data
        for (id, mark) in zip(items_ids, ratings):
            new_user_row[id] = mark

        logger.info('__init_new_row method successfully executed')
        return new_user_row

    def __create_indexer_dict(self, items_ids: list):
        """Create indexer for movie list

        Parameters
        ----------

        items_ids: list, list of movie order
        """

        indexer = {}
        for i, val in enumerate(items_ids):
            indexer[val] = i

        logger.info('__create_indexer_dict method successfully executed')
        return indexer

    def __sort_items_by_ids(self, items: pd.DataFrame, items_ids: list):
        """Method that sort movies in given dataframe
        by given list with ids

        Parameters
        ----------

        items: pd.DataFrame, movies dataframe
        items_ids: list, list of movie order
        """

        # create indexer
        indexer = self.__create_indexer_dict(items_ids)

        # get movies with id from items_ids
        items = items.loc[items['movie_id'].isin(items_ids), :]

        # change movies order
        items.loc[:, ['order']] = items['movie_id'].map(indexer)

        # get movies names
        names = items.sort_values('order')['title'].values.tolist()

        logger.info('__sort_items_by_ids method successfully executed')
        return names

    def __set_evaluate_results(self, rmse: float):
        """Method that renew information about current model

        Parameters
        ----------

        rmse: float, current calculated rmse
        """

        self.options.current_accuracy = rmse
        self.options.datetime_accuracy_test = datetime.now(pytz.timezone("Asia/Barnaul")).strftime("%m-%d-%Y:::%H:%M:%S.%f:::UTC%Z")

    def get_info(self):
        """Method that return current model state

        Parameters
        ----------

        """

        if not os.path.exists(self.options.credentials_file):
            logger.error(f'credentials file is not found on path: {self.options.credentials_file}')
            raise FileNotFoundError('credentials file is not found')
        credentials_file = open(self.options.credentials_file, 'r')
        lines = credentials_file.readlines()
        creation_date, author = lines[0].strip(), lines[1].strip()

        info_dict = {
            'accuracy(rmse)' : self.options.current_accuracy,
            'time': self.options.datetime_accuracy_test,
            'docker_creation_time_datetime': creation_date,
            'credentials': author
        }

        return info_dict


class CliWrapper(object):
    """
    This is a wrapper class for model instance
    to use model with CLI commands
    """

    def __init__(self):
        logger.info('started CliWrapper instance initialization')
        self.options = BaseOptions()
        self.root = BaseModel(self.options)
        logger.info('CliWrapper instance successfully inited')

    def train(self, dataset: str = None):
        """Method that run standard train method

        Parameters
        ----------

        dataset : string, path to train dataset
        """

        self.root.train(dataset)
        logger.info('train method successfully executed')

    def evaluate(self, dataset: str = None):
        """Method that run standard train and
        evaluate methods and return calculated RMSE

        Parameters
        ----------

        dataset : string, path to test dataset, model will train with dataset
        that stored in default train dataset path
        """

        self.root.train()
        rmse = self.root.evaluate(dataset)
        logger.info('evaluate method successfully executed')

    def predict(self, dataset: str = None, amount: int = 5):
        """Method that predict top {amount} similar movies
        for every user in dataset


        Parameters
        ----------

        dataset : string, path to test dataset, model will train with dataset
        that stored in default train dataset path
        amount : int, amount of similar movies
        """

        self.root.train()
        output = self.root.predict_dataset(dataset, amount)
        print(output, 'predicted')
        logger.info('predict method successfully executed')

    def surprise_train(self, dataset: str = None):
        """Method that run sklearn-surprise based train method

        Parameters
        ----------

        dataset : string, path to train dataset
        """

        self.root.surprise_train(dataset)
        logger.info('surprise_train method successfully executed')

    def surprise_evaluate(self, dataset: str = None):
        """Method that run sklearn-surprise based train and
        evaluate methods and return calculated RMSE

        Parameters
        ----------

        dataset : string, path to test dataset, model will train with dataset
        that stored in default train dataset path
        """

        self.root.surprise_train()
        rmse = self.root.surprise_evaluate(dataset)
        print(rmse)
        logger.info('surprise_evaluate method successfully executed')


if __name__ == "__main__":

    fire.Fire(CliWrapper)
    logger.info('fire\' CliWrapper is running')
