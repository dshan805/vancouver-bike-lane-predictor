import sys
import os
import logging
import numpy as np
from datetime import datetime
import pandas as pd
from read_data import read_csv_from_directory, read_excel_from_url

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import bikedata_url, bikedata_sheet_name, weather_directory_path

current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# setting up logging 
log_dir = "logs" # Create logs folder if it doesn't exist
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'process_data_{current_datetime}.log')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=log_file,
                    filemode='a')

# create a directory to save processed data
os.makedirs('data/processed', exist_ok=True)

def preprocess_bike_data(bike_data, threshold=0.7):
    try:
        logging.info("Starting preprocessing bike data")
        bike_data['date'] = pd.to_datetime(bike_data['date'])
        bike_data['Year'] = bike_data['date'].dt.year
        bike_data = bike_data.groupby(['date', 'Location'], as_index=False).sum()
        bike_data['Volume'] = bike_data['Volume'].round()

        bike_data_missing = bike_data.pivot(index='date', columns='Location', values='Volume').reset_index()
        zeros_count_perc = ((bike_data_missing == 0).sum() + bike_data_missing.isna().sum()) / len(bike_data_missing)
        cols_to_drop = zeros_count_perc[zeros_count_perc > threshold].index
        bike_data_cleaned = bike_data_missing.drop(columns=cols_to_drop)

        bike_data_no_outliers = replace_outliers(bike_data_cleaned)
        bike_data_no_outliers = bike_data_no_outliers.apply(replace_consecutive_missing_values, threshold=3)
        logging.info("Finished preprocessing bike data.")

        return bike_data_no_outliers
    except Exception as e:
        logging.error("Error during preprocessing bike data: {e}")
        return None

def replace_outliers(df):
    try:
        logging.info("Starting outlier replacement")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
        logging.info("Finished outlier replacement.")
        return df
    except Exception as e:
        logging.error(f"Error during outlier replacement: {e}")
        return None

def replace_consecutive_missing_values(series, threshold=3):
    try:
        logging.info("Starting consecutive missing values replacement")
        count = 0
        indices_to_nan = []
        for i, value in enumerate(series):
            if value == 0 or pd.isna(value) or value == '':
                count += 1
            else:
                if count > threshold:
                    indices_to_nan.extend(range(i - count, i))
                count = 0
        if count > threshold:
            indices_to_nan.extend(range(len(series) - count, len(series)))
        series.iloc[indices_to_nan] = np.nan
        logging.info("Finished replacement of consecutive missing values.")
        return series
    except Exception as e:
        logging.error(f"Error during consecutive missing values replacement: {e}")
        return None

def preprocess_weather_data(weather_data):
    try:
        logging.info("Starting preprocessing weather data")
        weather_data = weather_data.drop(columns=weather_data.columns[weather_data.isna().sum()/len(weather_data) > 0.7])
        # select columns of interest 
        weather_dropcols = ['Longitude (x)', 'Latitude (y)', 'Station Name', 'Dir of Max Gust (10s deg)', 'Dir of Max Gust Flag',
            'Spd of Max Gust (km/h)', 'Spd of Max Gust Flag'] # we are dropping gust b/c it's missing a lot of data
        weather_data = weather_data.drop(columns=weather_dropcols)
        weather_data = weather_data[weather_data['Date/Time'] <= '2024-10-24']
        weather_data = weather_data.dropna()   # drop missing data  
        logging.info("Finished preprocessing weather data.")
        return weather_data
    except Exception as e:
        logging.error(f"Error during preprocessing weather data: {e}")
        return None

# join weather data and bike data 
def join_data(bike_data, weather_data):
    try:
        logging.info("Starting joining bike and weather data")
        weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'], errors='coerce')
        # join the data
        bike_weather = pd.merge(bike_data, weather_data, how='left',
                            left_on='date', right_on='Date/Time')
        
        # drop rows with missing Climate ID
        bike_weather = bike_weather.dropna(subset=['Climate ID'])

        # create a list of bike lane columns
        bikelane_cols = bike_data.columns[1:]

        # fix the data so that each row is date + bike lane
        id_vars = [col for col in bike_weather.columns.tolist()
                if col not in bikelane_cols]
        
        bike_weather_melt = pd.melt(bike_weather, id_vars=id_vars, 
                                var_name = 'bikelane', value_name='num_usage')
        
        bike_weather_melt['year/month'] = bike_weather_melt['Year'].astype(int).astype(str) + '/' + bike_weather_melt['Month'].astype(int).astype(str).str.zfill(2)
        bike_weather_melt['year/month'] = pd.to_datetime(bike_weather_melt['year/month'], format='%Y/%m').dt.strftime('%Y/%m')

        # add day of week to bike_weather_melt
        bike_weather_melt['day_of_week'] = bike_weather_melt['date'].dt.day_name()
        # drop all values where num_usage is missing
        bike_weather_melt = bike_weather_melt.dropna(subset=['num_usage'])

        logging.info("Finished joining bike and weather data.")

        return bike_weather_melt
    except Exception as e:
        logging.error(f"Error during joining bike and weather data: {e}")
        return None
    
def get_bike_weather():
    """Reads, processes, and joins bike and weather data into a DataFrame."""
    try:
        bike_data = read_excel_from_url(bikedata_url, bikedata_sheet_name)
        weather_data = read_csv_from_directory(weather_directory_path)
        logging.info("Data read successfully.")
        if bike_data is not None and weather_data is not None:
            bike_data_cleaned = preprocess_bike_data(bike_data)
            weather_data_cleaned = preprocess_weather_data(weather_data)
            bike_weather = join_data(bike_data_cleaned, weather_data_cleaned)
            logging.info("Data joined successfully.")
            # Save to CSV for record-keeping
            output_path = f'data/processed/bike_weather_{current_datetime}.csv'
            bike_weather.to_csv(output_path, index=False)
            logging.info(f"Data written to {output_path}")
            
            return bike_weather
        else:
            logging.info("Error: Unable to read bike or weather data.")
            return None
    except Exception as e:
        logging.info(f"Error in get_bike_weather: {e}")
        return None
    
if __name__ == "__main__":
    try:
        bike_data = read_excel_from_url(bikedata_url, bikedata_sheet_name)
        weather_data = read_csv_from_directory(weather_directory_path)

        if bike_data is not None:
            bike_data_cleaned = preprocess_bike_data(bike_data)
            print(f"Bike data cleaned shape: {bike_data_cleaned.shape}")
            print(bike_data_cleaned.head())

        if weather_data is not None:
            weather_data_cleaned = preprocess_weather_data(weather_data)
            print(f"Weather data cleaned shape: {weather_data_cleaned.shape}")
            print(weather_data_cleaned.head())

        if bike_data is not None and weather_data is not None:
            bike_weather = join_data(bike_data_cleaned, weather_data_cleaned)
            print(f"Joined data shape: {bike_weather.shape}")
            print(bike_weather.head())
            bike_weather.to_csv(f'data/processed/bike_weather_{current_datetime}.csv', index=False)

    except Exception as e:
        print(f"Error during processing: {e}")