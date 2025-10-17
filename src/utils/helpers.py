import numpy as np
import random

def set_random_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    # For TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
