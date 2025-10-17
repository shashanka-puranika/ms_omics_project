# src/data/loading.py
import pyopenms
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class MassSpecDataLoader:
    """Load and preprocess mass spectrometry data from various formats."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_mzml(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from mzML file."""
        exp = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(file_path, exp)
        
        # Extract m/z and intensity arrays
        spectra = []
        for spec in exp:
            mz_array, intensity_array = spec.get_peaks()
            spectra.append((mz_array, intensity_array))
            
        return spectra
    
    def load_batch(self, file_pattern: str) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """Load multiple files matching a pattern."""
        files = list(self.data_dir.glob(file_pattern))
        data = {}
        
        for file_path in files:
            try:
                data[file_path.name] = self.load_mzml(str(file_path))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return data
