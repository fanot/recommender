import os
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

class BaseOptions:
    def __init__(self):
        logger.info('BaseOptions initialization started')

        # Determine the base working directory of the project
        self.core_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        # Default model configuration
        self.model_name = 'model'
        self.model_extension = 'csv'

        # Define default paths related to model and data storage
        self.model_store = os.path.join(self.core_directory, 'root', 'model', 'store')
        self.model_data_path = self._generate_model_data_path()

        # Define default paths for training and test data
        self.train_data_folder = os.path.join(self.core_directory, 'data', 'train')
        self.test_data_folder = os.path.join(self.core_directory, 'data', 'test')

        # Specific data file paths
        self.items_data_path = os.path.join(self.train_data_folder, 'movies.dat')
        self.users_data_path = os.path.join(self.train_data_folder, 'users.dat')
        self.train_data_path = os.path.join(self.train_data_folder, 'ratings_train.dat')
        self.test_data_path = os.path.join(self.test_data_folder, 'ratings_test.dat')

        # Configuration for SVD-based models
        self.n_vectors = 30

        # Data loading configuration
        self.data_loading_sep = '::'
        self.data_loading_engine = 'python'
        self.encoding = 'windows-1251'

        # Rating scale settings for surprise library
        self.rating_scale = (1, 5)

        # SVD-specific settings
        self.n_epochs = 20

        # Path to credentials file
        self.credentials_file = os.path.join(self.core_directory, 'credentials.txt')

        # Evaluation metrics (initialized as not evaluated)
        self.current_accuracy = 'Model hasn\'t been evaluated yet'
        self.datetime_accuracy_test = 'Model hasn\'t been evaluated yet'

        self._ensure_directories_exist()
        logger.info('BaseOptions instance successfully initialized')

    def _generate_model_data_path(self):
        """Generates the full path to the model data file."""
        return os.path.join(self.model_store, f'{self.model_name}.{self.model_extension}')

    def renew_model_name_and_path(self, model_name: str):
        """
        Updates the model's name and recalculates the model data path.

        Args:
            model_name (str): New name for the model.

        Returns:
            str: The new full path to the model data file.
        """
        logger.info('Renewing model name and path')
        self.model_name = model_name
        self.model_data_path = self._generate_model_data_path()
        logger.info('Model name and path renewed successfully')
        return self.model_data_path

    def _ensure_directories_exist(self):
        """Ensures that necessary directories exist, creating them if they don't."""
        directories = [self.model_store, self.train_data_folder, self.test_data_folder]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f'Created directory: {directory}')

