import pandas as pd
import numpy as np
import os

class DatasetLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def read_data(self):
        try:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"File not found: {self.filepath}")
            if not self.filepath.endswith('.csv'):
                raise ValueError("Only CSV files are supported.")

            self.data = pd.read_csv(self.filepath)
            self.numpy_data = self.data.to_numpy()
            print("Dataset loaded successfully.")
            print(self.data.head(20))
            return self.data

        except FileNotFoundError as fnf_error:
            print(f" Error: {fnf_error}")
        except pd.errors.EmptyDataError:
            print(" Error: The file is empty.")
        except pd.errors.ParserError:
            print(" Error: There was a problem parsing the CSV file.")
        except Exception as e:
            print(f" An unexpected error occurred: {e}")
