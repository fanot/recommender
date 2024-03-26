# Recommender System Based on Matrix Factorization

This project represents the implementation of a recommendation system based on matrix factorization performed on the 5th module of the 3rd year of HITS, Tomsk State University, Tomsk.

## Installation

### Within Docker

If you want to run the project in Docker, follow these instructions:

1. Enter into project's folder
2. Run docker composer
Now server is running and listening on 127.0.0.1:5000

### Without Docker

If you want to run the project without Docker:

1. Copy (or move) data and logs folders from basedir to ./root (inside docker volumes do that)
2. Run root/main.py
Now server is running on the same address

## CLI
foo@bar:$ python model.py train --dataset=/path/to/train/dataset

ex(python model/model.py train --dataset='data/train/ratings_train.dat' ) 

foo@bar:$ python model.py evaluate --dataset=/path/to/evaluation/dataset

ex(python model/model.py evaluate --dataset='data/test/ratings_test.dat' trained) 

foo@bar:~$ python model.py predict --dataset=/path/to/evaluation/dataset

ex(python model/model.py predict --dataset='data/test/ratings_test.dat' trained)

Note: use from cli branch, also it works with surprise(sirprise_evaluate, etc)
## Fanot's API

Endpoints:

/api/predict: 
Receives a list [[movie_name_1, movie_name_2, .., movie_name_N ], [rating_1, rating_2, .., rating_N]] and returns TOP M (default 20, also a parameter) recommended movies with corresponding estimated rating. Sort descending. 
Example:
{
"movies_ratings": [
["Inception", "The Dark Knight", "Interstellar"],
[4, 4, 4]
]
}

/api/log: 
Last 20 rows of log.

/api/info: 
Service Information: Your Credentials, Date and time of the build of the Docker image, Date, time and metrics of the training of the currently deployed model.

/api/reload: 
Reload the model.

/api/similar: 
Returns a list of similar movies. 
Example:
{
"movie_name": "Rushmore",
"N": 5
}

/api/surprise_evaluate: 
Evaluate sklearn-surprise based model and renew best accuracy.


## Sources

- [Recommender System Based on Matrix Factorization](https://www.linkedin.com/pulse/fundamental-matrix-factorization-recommender-system-saurav-kumar#:~:text=Matrix%20factorization%20is%20an%20extensively,users%20might%20be%20interested%20in.)

