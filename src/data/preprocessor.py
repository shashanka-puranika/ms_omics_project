import pandas as pd
import numpy as np
from pathlib import Path
from pyteomics import parser, mass
from collections import Counter
from typing import List, Dict, Tuple

class LabelGenerator:
    """Generates labels for elements and molecules from peptide identifications."""
    def __init__(self, config):
        self.raw_dir = Path(config['paths']['raw_data_dir'])
        self.processed_dir = Path(config['paths']['processed_data_dir'])
        self.pep_file_path = self.raw_dir / "PXD001819" / "peptides.txt"
        self.elements_to_predict = ['C', 'H', 'N', 'O', 'S']

    def _calculate_element_composition(self, peptide_sequence: str) -> Dict[str, int]:
        """Calculates elemental composition of a peptide."""
        # pyteomics mass.calculate_mass can give composition
        # We'll use a more direct approach for clarity
        composition = Counter()
        # Add elements from amino acids
        for aa in peptide_sequence:
            composition.update(parser.mass Composition(aa))
        # Add water molecule for the full peptide
        composition.update({'H': 2, 'O': 1})
        return {el: composition.get(el, 0) for el in self.elements_to_predict}

    def _load_peptide_ids(self) -> pd.DataFrame:
        """Loads the peptide identification file."""
        if not self.pep_file_path.exists():
            raise FileNotFoundError(f"Peptide ID file not found at {self.pep_file_path}")
        
        # MaxQuant peptides.txt is tab-separated
        df = pd.read_csv(self.pep_file_path, sep='\t')
        # We only need the sequence and the file it was found in
        # The 'Raw file' column in the dataset corresponds to our mzML file names
        df = df[['Sequence', 'Raw file']].dropna()
        df['Raw file'] = df['Raw file'].apply(lambda x: x.replace('.raw', '.mzML'))
        return df

    def generate_labels(self, spectra_data: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates element and molecule labels for each spectrum."""
        peptide_df = self._load_peptide_ids()
        
        # Create a mapping from file name to its identified peptides
        file_to_peptides = peptide_df.groupby('Raw file')['Sequence'].apply(list).to_dict()
        
        element_labels = []
        molecule_labels = []

        # Get all unique peptides for molecule labeling
        all_unique_peptides = sorted(list(set(peptide_df['Sequence'])))

        for spec in spectra_data:
            file_name = spec['file_name']
            spectrum_id = spec['spectrum_id']
            
            # Default labels are all zeros
            el_label = {el: 0 for el in self.elements_to_predict}
            mol_label = {pep: 0 for pep in all_unique_peptides}

            if file_name in file_to_peptides:
                peptides_in_file = file_to_peptides[file_name]
                
                # For simplicity, we assume any peptide identified in the file
                # could be present in any of its spectra. This is a weak assumption
                # but necessary without spectrum-peptide matching (e.g., from mzIdentML).
                # A better approach would be to use an identification file that maps
                # PSMs (Peptide Spectrum Matches) to specific scan numbers.
                
                # For this demo, we'll label the spectrum with ALL peptides found in its file.
                for pep_seq in peptides_in_file:
                    if pep_seq in mol_label:
                        mol_label[pep_seq] = 1
                        # Add elemental composition
                        comp = self._calculate_element_composition(pep_seq)
                        for el, count in comp.items():
                            el_label[el] = 1 # Mark element as present

            element_labels.append({'spectrum_id': spectrum_id, **el_label})
            molecule_labels.append({'spectrum_id': spectrum_id, **mol_label})

        element_df = pd.DataFrame(element_labels)
        molecule_df = pd.DataFrame(molecule_labels)

        # Save processed labels
        element_df.to_csv(self.processed_dir / "element_labels.csv", index=False)
        molecule_df.to_csv(self.processed_dir / "molecule_labels.csv", index=False)
        
        print(f"Generated labels for {len(element_df)} spectra.")
        print(f"Number of unique molecules (peptides) to predict: {len(all_unique_peptides)}")

        return element_df, molecule_df
