import pandas as pd
import os
import logging
from datetime import datetime
import numpy as np 

np.int = int

# Set up logging
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_dir = "logs" # Create logs folder if it doesn't exist
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'process_data_{current_time}.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=log_file,
                    filemode='a')

def read_csv_from_directory(directory_path):
    """Read and combine weather data from all CSV files in a directory."""
    try:
        csv_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]
        logging.info(f"Found {len(csv_files)} CSV files in directory: {directory_path}")
        weather_data = pd.concat([pd.read_csv(file) for file in csv_files], axis=0)
        logging.info("Successfully combined weather data from directory.")
        return weather_data
    except Exception as e:
        logging.error(f"Error reading or combining weather data: {e}")
        return None

def read_excel_from_url(url, sheet_name):
    """Read an Excel file directly from a URL into a pandas DataFrame."""
    try:
        logging.info(f"Attempting to read Excel file from URL: {url}")
        data = pd.read_excel(url, sheet_name=sheet_name)
        logging.info(f"Successfully loaded sheet {sheet_name} from URL: {url}")
        return data
    except Exception as e:
        logging.error(f"Error reading Excel file from URL: {e}")
        return None

def main():
    # Example: Load and combine weather data from a directory
    weather_directory_path = 'data/raw'
    weather_data = read_csv_from_directory(weather_directory_path)
    if weather_data is not None:
        logging.info("Weather data loaded successfully.")
        print("Weather Data:")
        print(weather_data.head())

    # Example: Load bike data from an Excel file
    bike_excel_url = "https://vancouver.ca/files/cov/bike-volume-2021-2024.xlsx"
    bike_sheet_name = "City of Vancouver Bike Data"
    bike_data = read_excel_from_url(bike_excel_url, bike_sheet_name)
    if bike_data is not None:
        logging.info("Bike data loaded successfully.")
        print("Bike Data:")
        print(bike_data.head())

if __name__ == "__main__":
    main()
