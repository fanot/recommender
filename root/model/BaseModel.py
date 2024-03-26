from abc import ABC, abstractmethod
from model.options import BaseOptions

class BaseModel(ABC):
    def __init__(self, n_vectors: int = 5):
        """
        Initializes the BaseModel with optional configuration for SVD-based models.
        
        Args:
            n_vectors (int): The dimension of latent vectors for SVD-based models. For non-SVD models, this can be ignored or set to None.
        """
        self.n_vectors = n_vectors  # Dimension of latent vectors for SVD-based models; irrelevant for non-SVD models.
        self.data = None  # Placeholder for model's predictions data.

    @abstractmethod
    def load_data(self, options: BaseOptions):
        """
        Loads prediction data from a file into the model. This method must be implemented by subclasses.
        
        Args:
            options (BaseOptions): Configuration options containing model data path and other settings.
        """
        pass

    @abstractmethod
    def fit(self):
        """
        Trains the model on the loaded data. This is an abstract method that must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def save(self, name: str, options: BaseOptions):
        """
        Saves the model's prediction data to a file on disk. Implementing this method in subclasses allows for customized saving logic.
        
        Args:
            name (str): The name under which to save the model data.
            options (BaseOptions): Configuration options that may influence how and where the data is saved.
        """
        pass
