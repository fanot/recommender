import os
import logging
import numpy as np
from flask import Flask, request, jsonify, make_response
from threading import Lock
import sys
sys.path.append('/root\\webapp')

import json

from pydotplus import basestring
from webapp.service import Service

# lock instance to protect app from Race Conditions
LOCK = Lock()

# Service class instance to handle recieved data
service = Service()

# logger instance
logger = logging.getLogger(__name__)

# flask app instance
app = Flask(__name__)

# Add url prefix '/api'
# when run app in 'dev' mode
app.config['APPLICATION_ROOT'] = '/api'


@app.route('/predict', methods=['POST'])
def predict():
    """
    Method that predict movie ratings by given input and recommended movies with
    corresponding estimated rating sort descending

    inputs:
    movies_ratings : list([movie_name_1, movie_name_2, .., movie_name_N ], [rating_1, rating_2, .., rating_N])
    M: int (default 20)

    output:
    Response with JSON data:
        {'message': list([movie_name_1, movie_name_2, .., movie_name_M], [rating_1, rating_2, .., rating_M]])}
    """
    logger.info('reached /api/predict/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'Server is busy'}), 403)

    with LOCK:
        request_data = request.get_json()
        if request_data is None:
            return make_response(jsonify({'message': 'Request body doesn\'t contain any data'}), 400)
        try:
            movie_names_ratings = request_data['movies_ratings']
            movie_names = movie_names_ratings[0]
            ratings = movie_names_ratings[1]

            m = request_data.get('M', 20)  # Set default value for m if not provided in request_data
            if not isinstance(m, int) or not (1 <= m <= 25):
                m = 20

            if not isinstance(movie_names_ratings, list):
                return make_response(jsonify({'message': 'Given values are not double lists'}), 400)

            if len(movie_names_ratings) != 2:
                return make_response(jsonify({'message': 'Expected only two arrays'}), 400)

            if len(movie_names) != len(ratings):
                return make_response(jsonify({'message': 'Lists have different sizes'}), 400)

            if not is_list_of_strings(movie_names):
                return make_response(jsonify({'message': 'Movies names list consist not only of strings'}), 400)

            if not is_list_of_ints(ratings):
                return make_response(jsonify({'message': 'Movies names list consist not only of ints'}), 400)
        except KeyError as e:
            logger.error(e)
            return make_response(jsonify({'message': 'No \'movies_ratings\' key in JSON data'}), 400)
        except json.decoder.JSONDecodeError as e:
            logger.error(e)
            return make_response(jsonify({'message': 'Invalid JSON data'}), 400)

        try:
            result = service.predict([movie_names, ratings], m)
            logger.info('result successfully predicted')
            return make_response(jsonify(result), 200)
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/log', methods=['GET'])
def log():
    """
    Method that return last 20 rows of log-file.

    inputs:
    None

    output:
    Responce with JSON data:
        {'message': list(log_row_1, log_row_2, ... , log_row_20)}
    """
    logger.info('reached /api/log/ endpoint')

    if LOCK.locked():
        return make_response(jsonify({'error': 'Server is busy'}), 403)
    with LOCK:
        try:
            output = service.log()
            logger.info('logs successfully received')
            return make_response(jsonify({'last 20 rows of logs': output}))
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/info', methods=['GET'])
def info():
    """
    Method that return current information about model

    inputs:
    None

    output:
    Responce with JSON data:
        {'message':
            'accuracy(rmse)' : float, # accuracy of the model
            'time': datetime, # time of last model's evaluation
            'docker_creation_time_datetime': datetime, # time of docker image build
            'credentials': string # name of the author of the model
        }
    """
    logger.info('reached /api/info/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'Server is busy'}), 403)
    with LOCK:
        try:
            info = service.info()
            logger.info('docker info successfully recieved')
            return make_response(jsonify({'message': info}))
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/reload', methods=['POST'])
def reload():
    """
    Method that reload model

    inputs:
    None

    output:
    Responce with JSON data:
        {'Result': 'model successfully reloaded' }
    """
    logger.info('reached /api/reload/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'server is busy'}), 403)
    with LOCK:
        try:
            service.reload()
            logger.info('model successfully reloaded')
            return make_response(jsonify({'Result': "model successfully reloaded"}))
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Method that evaluate model with default model

    inputs:
    None

    output:
    Responce with JSON data:
        {'Result': 'model successfully evaluated' }
    """
    logger.info('reached /api/evaluate/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'server is busy'}), 403)
    with LOCK:
        try:
            service.evaluate()
            logger.info('model successfully evaluated')
            return make_response(jsonify({'Result': "model successfully evaluated"}))
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/surprise_evaluate', methods=['POST'])
def surprise_evaluate():
    """
    Method that evaluate sklearn-surprise based model

    inputs:
    None

    output:
    Responce with JSON data:
        {'Result': 'model successfully evaluated' }
    """
    logger.info('reached /api/surprise_evaluate/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'server is busy'}), 403)
    with LOCK:
        try:
            service.surprise_evaluate()
            logger.info('model successfully evaluated, surprise')
            return make_response(jsonify({'Result': "model successfully evaluated surprise"}))
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


@app.route('/similar', methods=['POST'])
def similar():
    """
    Method that predict the most similair movies to given movie

    inputs:
    movie_name : string # name of the movie which we will be use to find similar
    N: int # amount of similar movies

    output:
    Responce with JSON data:
        {'message': list([movie_name_1, movie_name_2, .., movie_name_M], [rating_1, rating_2, .., rating_M]])}
    """
    logger.info('reached /api/similar/ endpoint')
    if LOCK.locked():
        return make_response(jsonify({'error': 'Server is busy'}), 403)
    with LOCK:
        request_data = request.get_json()
        if request_data is None:
            return make_response(jsonify({'message': 'Request body doesn\'t contain JSON data'}), 400)
        try:
            n = 5
            movie_name = request_data['movie_name']
            if 'N' in request_data:
                n = request_data['N']
                if not isinstance(n, int) or (isinstance(n, int) and n > 50):
                    n = 5
            if not isinstance(movie_name, str) or movie_name == '':
                return make_response(jsonify({'message': 'Wrong \'movie_name\' value'}), 400)
        except KeyError as e:
            logger.error(e)
            return make_response(jsonify({'message': 'No \'movie_name\' key in JSON data'}), 400)
        except json.decoder.JSONDecodeError as e:
            logger.error(e)
            return make_response(jsonify({'message': 'Invalid JSON data'}), 400)
        try:
            output = service.similar(movie_name, n)
            logger.info('successfully recieved similair results')
            return make_response(jsonify(output), 200)
        except Exception as e:
            logger.error(e)
            return make_response(jsonify({'error': 'Something went wrong'}), 500)


def is_list_of_strings(lst):
    """
    Method that check if given list consist only of strings

    inputs:
    lst : list

    output:
    True - if only strings in list / False if not
    """
    return bool(lst) and isinstance(lst, list) and all(isinstance(elem, basestring) for elem in lst)


def is_list_of_ints(lst):
    """
    Method that check if given list consists of only of int

    inputs:
    lst : list

    output:
    True - if only strings in int / False if not
    """
    return all(isinstance(x, int) for x in lst)
