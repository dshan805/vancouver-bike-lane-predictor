# Vancouver Bike Lane Usage Predictor

Check out the deployed Streamlit application: [Vancouver Bike Lane Predictor App](https://dshan805-vancouver-bike-lane-predictor-app-8lddat.streamlit.app/)

This project predicts bike lane usage in Vancouver, focusing on the impact of weather conditions. Using a machine learning model (Generative Additive Model), it forecasts daily bike lane activity based on weather data to support infrastructure planning and assess the year-round viability of reallocating vehicle lanes for bike lanes in a city with high precipitation.

## Features

- **Predictive Model**: A Generalized Additive Model (GAM) trained on bike lane and weather data.
- **Streamlit Application**: Interactive interface to visualize predictions and explore relationships between weather and bike usage.
- **Data Preprocessing**: Automated cleaning and integration of bike and weather datasets.

## Objectives

- Forecast bike lane usage under varying weather conditions.
- Provide insights to guide infrastructure planning and evaluate the effectiveness of bike lanes in Vancouver's climate.

## Data Sources

- **Bike Lane Usage Data**: Bikelane usage statistics across various Vancouver locations (2021–2024).
- **Weather Data**: Metrics including temperature, precipitation, and snowfall collected daily.

Both datasets were preprocessed to remove outliers, handle missing values, and add relevant features like day-of-week dummy variables and interaction terms.

## Methodology

1. **Data Preprocessing**:  
   - Cleaned and aggregated bike and weather data.  
   - Removed outliers and imputed missing values.  
   - Engineered interaction terms to capture complex relationships between weather and bike usage.

2. **Predictive Modeling**:  
   - Trained Generalized Additive Models (GAMs) for each bike lane to forecast usage.  
   - Evaluated model performance using Mean Squared Error (MSE).

3. **Visualization**:  
   - Built an interactive Streamlit app to explore predictions and visualize relationships.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/dshan805/vancouver-bike-lane-predictor.git
   cd vancouver-bike-lane-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Access the app in your browser at `http://localhost:8501`.

## Directory Structure

```
├── data/
│   ├── raw/              # Raw weather data
│   ├── processed/        # Processed data files
├── models/               # Trained GAM models
├── logs/                 # Log files 
├── app.py                # Streamlit application
├── scripts/
│   ├── process_data.py   # Data preprocessing scripts
│   ├── train_model.py    # Model training scripts
│   ├── read_data.py      # Read in data  
├── requirements.txt      # Python dependencies
├── config.py             # Configuration file for dataset paths and settings
└── README.md             # Project documentation
```
## Related Work

For an analysis of the bike lane usage data, visit the repository at: [Bike Lane Usage Analysis](https://github.com/dshan805/bike_lane_usage)

## Future Enhancements

- Integrate real-time weather data for live predictions.
- Expand the model to include additional factors like holidays and traffic congestion.
- Enhance visualization with historical trends and confidence intervals for predictions.

