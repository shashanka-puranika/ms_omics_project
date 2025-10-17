import pyopenms
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

class MassSpecDataLoader:
    """Loads mass spectrometry data from mzML files."""
    def __init__(self, config):
        self.raw_dir = Path(config['paths']['raw_data_dir'])
        self.mz_range = config['feature_engineering']['mz_range']

    def load_mzml(self, file_path: str) -> List[Dict]:
        """Loads a single mzML file and extracts MS1 spectra."""
        exp = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(file_path, exp)
        
        spectra = []
        for i, spec in enumerate(exp):
            # We only want MS1 spectra for this project
            if spec.getMSLevel() != 1:
                continue
            
            mz_array, intensity_array = spec.get_peaks()
            
            # Filter by m/z range
            mask = (mz_array >= self.mz_range[0]) & (mz_array <= self.mz_range[1])
            mz_array = mz_array[mask]
            intensity_array = intensity_array[mask]

            if len(mz_array) == 0:
                continue

            spectra.append({
                'spectrum_id': f"{Path(file_path).stem}_{i}",
                'mz_array': mz_array,
                'intensity_array': intensity_array,
                'file_name': Path(file_path).name
            })
        return spectra

    def load_all_mzml_files(self) -> List[Dict]:
        """Finds and loads all mzML files in the raw data directory."""
        all_spectra = []
        mzml_files = list(self.raw_dir.glob('*.mzML'))
        
        if not mzml_files:
            raise FileNotFoundError("No .mzML files found in data/raw/. Run 'python main.py --mode download' first.")
            
        for file_path in mzml_files:
            print(f"Loading {file_path.name}...")
            spectra = self.load_mzml(str(file_path))
            all_spectra.extend(spectra)
        
        return all_spectra
