# src/features/feature_engineering.py
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from pyteomics import mass

class FeatureEngineer:
    """Engineer features from mass spectrometry peaks for ML models."""
    
    def __init__(self, mz_bins: int = 1000, mz_range: Tuple[float, float] = (50, 2000)):
        self.mz_bins = mz_bins
        self.mz_range = mz_range
        self.bin_edges = np.linspace(mz_range[0], mz_range[1], mz_bins + 1)
        
    def binned_spectrum(self, peak_mz: np.ndarray, peak_intensity: np.ndarray) -> np.ndarray:
        """Convert peaks to binned spectrum representation."""
        binned = np.zeros(self.mz_bins)
        
        # Assign peaks to bins
        bin_indices = np.digitize(peak_mz, self.bin_edges) - 1
        
        # Handle edge cases
        valid_indices = (bin_indices >= 0) & (bin_indices < self.mz_bins)
        bin_indices = bin_indices[valid_indices]
        peak_intensity = peak_intensity[valid_indices]
        
        # Sum intensities in each bin
        np.add.at(binned, bin_indices, peak_intensity)
        
        # Normalize
        if np.max(binned) > 0:
            binned = binned / np.max(binned)
            
        return binned
    
    def elemental_composition_features(self, peak_mz: np.ndarray, peak_intensity: np.ndarray) -> Dict:
        """Extract features related to elemental composition."""
        features = {}
        
        # Top N peaks
        top_n = 10
        if len(peak_mz) > 0:
            # Sort by intensity
            sorted_indices = np.argsort(peak_intensity)[::-1]
            top_mz = peak_mz[sorted_indices[:top_n]]
            top_intensity = peak_intensity[sorted_indices[:top_n]]
            
            # Calculate possible elemental compositions for top peaks
            for i, (mz, intensity) in enumerate(zip(top_mz, top_intensity)):
                # This is a simplified approach - in practice, you'd use more sophisticated methods
                features[f'top_{i+1}_mz'] = mz
                features[f'top_{i+1}_intensity'] = intensity
                
                # Estimate possible elemental composition (simplified)
                # In practice, you'd use algorithms like the Seven Golden Rules or SIRIUS
                features[f'top_{i+1}_possible_C'] = int(mz / 12.0)  # Simplified
                features[f'top_{i+1}_possible_H'] = int(mz / 1.0)   # Simplified
        else:
            # Handle empty spectra
            for i in range(top_n):
                features[f'top_{i+1}_mz'] = 0
                features[f'top_{i+1}_intensity'] = 0
                features[f'top_{i+1}_possible_C'] = 0
                features[f'top_{i+1}_possible_H'] = 0
                
        return features
    
    def molecular_fingerprint_features(self, peak_mz: np.ndarray, peak_intensity: np.ndarray) -> np.ndarray:
        """Generate molecular fingerprint-like features from peaks."""
        # Create a fixed-size fingerprint based on m/z patterns
        fingerprint = np.zeros(512)  # 512-bit fingerprint
        
        if len(peak_mz) > 0:
            # Normalize m/z to [0, 1] range
            normalized_mz = (peak_mz - self.mz_range[0]) / (self.mz_range[1] - self.mz_range[0])
            
            # Map to fingerprint indices
            indices = (normalized_mz * 511).astype(int)
            indices = np.clip(indices, 0, 511)
            
            # Set fingerprint bits based on intensity
            for idx, intensity in zip(indices, peak_intensity):
                if idx < 512:
                    fingerprint[idx] = min(1.0, fingerprint[idx] + intensity / np.max(peak_intensity))
                    
        return fingerprint
    
    def extract_features(self, spectra_data: List[Dict]) -> pd.DataFrame:
        """Extract all features from processed spectra data."""
        all_features = []
        
        for spectrum in spectra_data:
            peak_mz = spectrum['peak_mz']
            peak_intensity = spectrum['peak_intensity']
            
            # Extract different feature types
            binned = self.binned_spectrum(peak_mz, peak_intensity)
            elemental = self.elemental_composition_features(peak_mz, peak_intensity)
            fingerprint = self.molecular_fingerprint_features(peak_mz, peak_intensity)
            
            # Combine features
            features = {
                'spectrum_id': spectrum['spectrum_id'],
                'num_peaks': spectrum['num_peaks'],
                **{f'binned_{i}': val for i, val in enumerate(binned)},
                **{f'fp_{i}': val for i, val in enumerate(fingerprint)},
                **elemental
            }
            
            all_features.append(features)
            
        return pd.DataFrame(all_features)
