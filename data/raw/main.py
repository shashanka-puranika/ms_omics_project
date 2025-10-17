import argparse
import yaml
import os
from pathlib import Path

from src.data.download_data import DataDownloader
from src.data.loader import MassSpecDataLoader
from src.data.preprocessor import LabelGenerator
from src.features.feature_engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.utils.helpers import set_random_seed

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Mass Spec ML Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['download', 'train'],
                        help="Pipeline mode: 'download' or 'train'")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Create necessary directories
    Path(config['paths']['raw_data_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['processed_data_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['models_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['figures_dir']).mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    set_random_seed(config['general']['random_seed'])

    if args.mode == 'download':
        print("--- Starting Data Download ---")
        downloader = DataDownloader(config)
        downloader.run()
        print("--- Data Download Complete ---")

    elif args.mode == 'train':
        print("--- Starting Model Training Pipeline ---")
        
        # 1. Load Data
        print("Step 1: Loading raw MS data...")
        loader = MassSpecDataLoader(config)
        spectra_data = loader.load_all_mzml_files()
        print(f"Loaded {len(spectra_data)} spectra.")

        # 2. Generate Labels
        print("Step 2: Generating labels from peptide identifications...")
        label_gen = LabelGenerator(config)
        element_labels, molecule_labels = label_gen.generate_labels(spectra_data)
        print("Labels generated.")

        # 3. Feature Engineering
        print("Step 3: Engineering features from spectra...")
        feature_eng = FeatureEngineer(config)
        features_df = feature_eng.extract_features(spectra_data)
        print(f"Generated feature matrix with shape: {features_df.shape}")

        # 4. Train Models
        print("Step 4: Training models...")
        trainer = ModelTrainer(config)
        trainer.train_element_model(features_df, element_labels)
        trainer.train_molecule_model(features_df, molecule_labels)
        print("--- Model Training Pipeline Complete ---")

if __name__ == '__main__':
    main()
