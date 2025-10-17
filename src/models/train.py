from .element_predictor import ElementPredictor
from .molecule_predictor import MoleculePredictor

class ModelTrainer:
    """Orchestrates the training of all models."""
    def __init__(self, config):
        self.config = config
        self.element_predictor = ElementPredictor(config)
        self.molecule_predictor = MoleculePredictor(config)

    def train_element_model(self, features_df, element_labels_df):
        """Trains the element prediction model."""
        self.element_predictor.train(features_df, element_labels_df)

    def train_molecule_model(self, features_df, molecule_labels_df):
        """Trains the molecule prediction model."""
        self.molecule_predictor.train(features_df, molecule_labels_df)
