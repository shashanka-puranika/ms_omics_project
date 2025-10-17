import os
import requests
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET

class DataDownloader:
    """Downloads and extracts data for the PXD001819 dataset."""
    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config['paths']['raw_data_dir'])
        self.base_url = config['download']['base_url']
        self.files_to_download = config['download']['mzml_files']
        self.pep_id_file = config['download']['peptide_id_file']

    def download_file(self, url, dest_path):
        """Downloads a file with a progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(dest_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=dest_path.name
        ) as bar:
            for data in response.iter_content(block_size=1024):
                bar.update(len(data))
                f.write(data)

    def extract_gz(self, gz_path, dest_path):
        """Extracts a .gz file."""
        with gzip.open(gz_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def get_peptide_ids_from_zip(self):
        """PXD001819 contains peptides.txt in a zip. We need to find it."""
        # The actual file is in a zip. Let's find the zip URL.
        # The `mqpar.xml` is not the peptide file, the `peptides.txt` is.
        # We will manually point to the correct zip file for this project.
        zip_url = f"{self.base_url}/PXD001819.zip"
        zip_path = self.raw_dir / "PXD001819.zip"
        
        if not zip_path.exists():
            print(f"Downloading {zip_url}...")
            self.download_file(zip_url, zip_path)
        
        # Extract the zip
        print("Extracting zip file...")
        shutil.unpack_archive(zip_path, self.raw_dir)
        print("Zip file extracted.")
        
        # The peptides.txt is inside the extracted directory
        pep_file_path = self.raw_dir / "PXD001819" / "peptides.txt"
        if not pep_file_path.exists():
            raise FileNotFoundError(f"Could not find peptides.txt at {pep_file_path}")
        
        return pep_file_path

    def run(self):
        """Main download and extraction logic."""
        # Download and extract mzML files
        for file_rel_path in self.files_to_download:
            gz_url = f"{self.base_url}/{file_rel_path}"
            gz_filename = Path(file_rel_path).name
            gz_path = self.raw_dir / gz_filename
            mzml_path = self.raw_dir / gz_filename.replace('.gz', '')

            if not mzml_path.exists():
                if not gz_path.exists():
                    print(f"Downloading {gz_url}...")
                    self.download_file(gz_url, gz_path)
                
                print(f"Extracting {gz_filename}...")
                self.extract_gz(gz_path, mzml_path)
                print(f"Extracted to {mzml_path}")
            else:
                print(f"{mzml_path} already exists. Skipping download.")
        
        # Get peptide identification file
        self.get_peptide_ids_from_zip()
        print("All data downloaded and prepared.")
