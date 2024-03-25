import logging
import sys
sys.path.append('/root\\model')

from model.model import BaseModel
from model.options import BaseOptions
from logger import LOG_FILE_PATH

logger = logging.getLogger(__name__)

class Service:
    def __init__(self):
        """Initializes the Service class, setting up the options, model, and triggering initial model training."""
        logger.info('Initializing Service instance...')
        self.options = BaseOptions()
        self.model = BaseModel(self.options)
        self.model.train()
        logger.info('Service instance initialized successfully.')

    def train(self):
        """Triggers the model's training process."""
        logger.info('Starting training process...')
        self.model.train()
        logger.info('Training process completed.')

    def evaluate(self):
        """Evaluates the current model's performance."""
        logger.info('Starting evaluation process...')
        self.model.evaluate()
        logger.info('Evaluation process completed.')

    def predict(self, item_ratings: list, M: int = 5) -> list:
        """
        Predicts ratings for a list of items.

        Args:
            item_ratings (list): A list containing pairs of movie names and ratings.
            M (int): The number of similar movies to predict.

        Returns:
            list: A list of movie names with their estimated ratings.
        """
        logger.info('Starting prediction process...')
        output = self.model.predict(item_ratings, M)
        logger.info('Prediction process completed.')
        return output

    def reload(self):
        """Resets the model state for a new session."""
        logger.info('Starting model reload...')
        self.model.warmup()
        logger.info('Model reloaded successfully.')

    def similar(self, movie_name: str, n: int = 10) -> list:
        """
        Finds similar movies to a given movie name.

        Args:
            movie_name (str): The name of the movie to find similarities for.
            n (int): The number of similar movies to return.

        Returns:
            list: A list of movie names with their similarities.
        """
        logger.info('Finding similar movies...')
        output = self.model.get_similar_items(movie_name, n)
        logger.info('Similar movies found successfully.')
        return output

    def log(self, n_rows: int = 20) -> list:
        """
        Retrieves the last N rows from the log file.

        Args:
            n_rows (int): The number of log rows to return.

        Returns:
            list: The last N rows of the log file.
        """
        logger.info('Retrieving log entries...')
        with open(LOG_FILE_PATH, "r", encoding='windows-1251') as log_file:
            log_rows = log_file.readlines()[-n_rows:]
        logger.info('Log entries retrieved successfully.')
        return log_rows

    def surprise_evaluate(self):
        """Trains and evaluates the model using a surprise method."""
        logger.info('Starting surprise evaluation...')
        self.model.surprise_train()
        self.model.surprise_evaluate()
        logger.info('Surprise evaluation completed successfully.')

    def info(self) -> dict:
        """
        Retrieves information about the current state of the model.

        Returns:
            dict: Information about the model's state.
        """
        logger.info('Retrieving model information...')
        info = self.model.get_info()
        logger.info('Model information retrieved successfully.')
        return info
