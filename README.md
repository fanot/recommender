Recommender System Based on Matrix Factorization
This project represents the implementation of a recommendation system based on matrix factorization performed on the 5th module of the 3rd year of HITS, Tomsk State University, Tomsk.

Installation
Within Docker
If you want to run project in Docker - follow these instructions:

1. Enter into project's folder
   \$ cd /path/to/recsys-mf/
2. Run docker compose*
   \$ docker-compose up --build
  
  Or if you want to run in detach mode:
  \$ docker-compose up --detach --build
Now server is running and listening on 127.0.0.1:5000

Without Docker
If you want to run project without Docker:

1. Copy(or move) data and logs folders from basedir to ./recsys (inside docker volumes do that)
2. run root/main.py
Now server is running on the same address

CLI
foo@bar:~$ python model.py train --dataset=/path/to/train/dataset
ex(python model/model.py train --dataset='data/train/ratings_train.dat'
)
foo@bar:~$ python model.py evaluate --dataset=/path/to/evaluation/dataset
ex(python model/model.py evaluate --dataset='data/test/ratings_test.dat'
trained)
foo@bar:~$ python model.py predict --dataset=/path/to/evaluation/dataset
ex(python model/model.py predict --dataset='data/test/ratings_test.dat'
trained)
Fanot`s api
Endpoints:

/api/predict. Recieves list [[movie_name_1, movie_name_2, .., movie_name_N ], [rating_1, rating_2, .., rating_N]] and returns TOP M (default 20, also a parameter) recommended movies with corresponding estimated rating. Sort descending. [[movie_name_1, movie_name_2, .., movie_name_M], [rating_1, rating_2, .., rating_M]]
example -{ "movies_ratings": [
        ["Inception", "The Dark Knight", "Interstellar"],
        [4, 4, 4]
    ]}
/api/log. Last 20 rows of log.
/api/info. Service Information: Your Credentials, Date and time of the build of the Docker image, Date, time and metrics of the training of the currently deployed model.
/api/reload. Reload the model.
/api/similar. returns list of similar movies {
  "movie_name": "Rushmore",
  "N": 5
}
api/surprise_evaluate. evaluate sklearn-surprise based model and renew best accuracy.

Sources
Рекомендательная система на основе SVD разложения матриц
Recommender System — Matrix Factorization
