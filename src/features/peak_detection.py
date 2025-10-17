# src/features/peak_detection.py
import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, List, Dict

class PeakDetector:
    """Detect peaks in mass spectrometry data."""
    
    def __init__(self, min_height: float = 0.01, prominence: float = 0.01):
        self.min_height = min_height
        self.prominence = prominence
        
    def detect_peaks(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect peaks in a single spectrum."""
        # Normalize intensity
        normalized_intensity = intensity_array / np.max(intensity_array)
        
        # Find peaks
        peaks, properties = find_peaks(
            normalized_intensity, 
            height=self.min_height,
            prominence=self.prominence
        )
        
        # Extract peak m/z and intensity values
        peak_mz = mz_array[peaks]
        peak_intensity = intensity_array[peaks]
        
        return peak_mz, peak_intensity
    
    def process_spectra(self, spectra: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
        """Process multiple spectra and extract peak information."""
        results = []
        
        for i, (mz, intensity) in enumerate(spectra):
            peak_mz, peak_intensity = self.detect_peaks(mz, intensity)
            
            results.append({
                'spectrum_id': i,
                'peak_mz': peak_mz,
                'peak_intensity': peak_intensity,
                'num_peaks': len(peak_mz)
            })
            
        return results
