import os
import logging
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from model.BaseModel import BaseModel
from model.options import BaseOptions

logger = logging.getLogger(__name__)

class SvdModel(BaseModel):
    def __init__(self, n_vectors: int = 5):
        """
        Initializes the SvdModel instance with the number of vectors for the SVD.

        Args:
            n_vectors (int): The number of singular values and vectors to compute.
        """
        self.n_vectors = n_vectors
        self.data = None
        logger.info('SvdModel instance successfully inited')

    def load_data(self, options: BaseOptions) -> None:
        """
        Loads data from the specified path in the options. Supports CSV files.

        Args:
            options (BaseOptions): Configuration options including the model data path.

        Raises:
            ValueError: If the file extension is not supported.
        """
        logger.info(f'Started load_data method, data_path: {options.model_data_path}')
        model_path_split = os.path.splitext(options.model_data_path)
        options.model_name = model_path_split[0]
        options.model_extension = model_path_split[1][1:]

        if options.model_extension.lower() == 'csv':
            self.data = pd.read_csv(options.model_data_path, encoding=options.encoding)
            self.data.columns = [int(x) for x in self.data.columns]
        else:
            logger.error(f"Unsupported model extension: {options.model_extension}")
            raise ValueError(f"Unsupported model extension: {options.model_extension}")
        logger.info('load_data method successfully executed')

    def fit(self, matrix: pd.DataFrame, n_vectors: int, mean_user_rating: np.ndarray, std_user_rating: np.ndarray) -> pd.DataFrame:
        """
        Fits the model using SVD on the provided matrix, applying normalization with the given mean and standard deviation of user ratings.

        Args:
            matrix (pd.DataFrame): The user-item ratings matrix.
            n_vectors (int): The number of singular values and vectors to compute.
            mean_user_rating (np.ndarray): The mean rating for each user.
            std_user_rating (np.ndarray): The standard deviation of ratings for each user.

        Returns:
            pd.DataFrame: The predicted ratings matrix.
        """
        logger.info('Started fit method')
        u, sigma, vt = svds(matrix.values, k=n_vectors)
        sigma_diag_matrix = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(u, sigma_diag_matrix), vt) * std_user_rating + mean_user_rating
        self.data = pd.DataFrame(predicted_ratings, columns=matrix.columns)
        logger.info('Fit method successfully executed')
        return self.data

    def save(self, name: str, options: BaseOptions) -> None:
        """
        Saves the model data to a CSV file specified in the options.

        Args:
            name (str): The name to save the model data under, altering options if different.
            options (BaseOptions): Configuration options, including the model save path.
        """
        logger.info('Started save method')
        if name != options.model_name:
            options.renew_model_name_and_path(name)
        self.data.to_csv(options.model_data_path, index=False)
        logger.info('Save method successfully executed')
