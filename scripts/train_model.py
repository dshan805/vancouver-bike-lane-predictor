import os
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pygam import LinearGAM, s, f
from process_data import get_bike_weather
import matplotlib.pyplot as plt
import numpy as np 

np.int = int

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory to save models
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

def train_and_save_models(bike_weather):
    """
    Train and save models for each bike lane, keeping all terms explicitly.
    """
    try:
        lane_results = {}

        # Add features to the dataset
        bike_weather['day_bike_interaction'] = bike_weather['day_of_week'].astype(str) + "_" + bike_weather['bikelane'].astype(str)
        bike_weather['bikelane_original'] = bike_weather['bikelane']
        bike_weather_melt_dummies = pd.get_dummies(bike_weather, columns=['day_of_week', 'bikelane'], drop_first=True)

        for lane in bike_weather_melt_dummies['bikelane_original'].unique():
            # Filter data for the specific bike lane and ensure it's a copy
            lane_data = bike_weather_melt_dummies[bike_weather_melt_dummies['bikelane_original'] == lane].copy()

            # Define features and target
            X = lane_data[['Mean Temp (°C)', 'Total Rain (mm)', 'Total Snow (cm)', 'Cool Deg Days (°C)'] +
                          [col for col in lane_data.columns if 'day_of_week' in col]].copy()
            y = lane_data['num_usage']

            # Create interaction terms between Mean Temp and day_of_week
            for day_col in [col for col in X.columns if 'day_of_week' in col]:
                X[f'MeanTemp_{day_col}'] = X['Mean Temp (°C)'] * X[day_col]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=42)

            # Train a GAM model with all terms explicitly defined
            gam = LinearGAM(
                s(0) +  # Smooth term for Mean Temp
                s(1) +  # Smooth term for Total Rain
                s(2) +  # Smooth term for Total Snow
                s(3) +  # Smooth term for Cool Deg Days
                f(4) +  # Fixed effect for day_of_week_Monday
                f(5) +  # Fixed effect for day_of_week_Saturday
                f(6) +  # Fixed effect for day_of_week_Sunday
                f(7) +  # Fixed effect for day_of_week_Thursday
                f(8) +  # Fixed effect for day_of_week_Tuesday
                f(9) +  # Fixed effect for day_of_week_Wednesday
                s(10) +  # Interaction term for MeanTemp_day_of_week_Monday
                s(11) +  # Interaction term for MeanTemp_day_of_week_Saturday
                s(12) +  # Interaction term for MeanTemp_day_of_week_Sunday
                s(13) +  # Interaction term for MeanTemp_day_of_week_Thursday
                s(14) +  # Interaction term for MeanTemp_day_of_week_Tuesday
                s(15)    # Interaction term for MeanTemp_day_of_week_Wednesday
            )
            gam.gridsearch(X_train, y_train)

            # Evaluate the model
            y_pred = gam.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            lane_results[lane] = mse

            # Save the model
            model_path = os.path.join(model_dir, f"{lane}_gam_model.pkl")
            with open(model_path, "wb") as model_file:
                pickle.dump(gam, model_file)

            # Convert X_test and y_test back to pandas objects before saving
            X_test_df = pd.DataFrame(X_test, columns=X.columns)  # Add column names from X
            y_test_series = pd.Series(y_test, name='num_usage')  # Add a name for the Series

            # Save X_test and y_test as pickle files
            X_test_path = os.path.join(model_dir, f"{lane}_X_test.pkl")
            y_test_path = os.path.join(model_dir, f"{lane}_y_test.pkl")
            X_test_df.to_pickle(X_test_path)
            y_test_series.to_pickle(y_test_path)
            
            logging.info(f"Trained and saved model for lane {lane} with MSE: {mse}")

        return lane_results

    except Exception as e:
        logging.error(f"Error during training and saving models: {e}")
        return None

# Example Usage
if __name__ == "__main__":
    bike_weather = get_bike_weather()

    if bike_weather is not None:
        # Train and save models
        results = train_and_save_models(bike_weather)
        if results:
            for lane, mse in results.items():
                print(f"Lane: {lane}, MSE: {mse}")
    else:
        logging.error("Failed to process bike_weather data.")
