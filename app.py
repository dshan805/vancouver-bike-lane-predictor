import streamlit as st
import pandas as pd
import altair as alt
import joblib
import sys
import os
import numpy as np

np.int = int

# Add scripts folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from process_data import get_bike_weather

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Add a title to the app
st.title("City of Vancouver - Bikelane Usage Predictor")

# Load and preprocess the data
@st.cache_data
def load_data():
    return get_bike_weather()

bike_weather_melt = load_data()

# Create layout columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Filter Options and Prediction Inputs")

    # Filter for bike lane selection
    selected_bikelane = st.selectbox(
        "Select a Bike Lane:",
        options=bike_weather_melt['bikelane'].unique(),
        index=0
    )

    # Prediction inputs
    mean_temp = st.number_input("Mean Temperature (°C):", value=15.0)
    total_precip = st.number_input("Total Precipitation (mm):", value=0.0)
    total_snow = st.number_input("Total Snow (cm):", value=0.0)
    day_of_week = st.selectbox("Day of Week:", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=0)

    cool_deg_days = mean_temp - 18  # Calculate Cool Degree Days

    # Prediction button
    if st.button("Predict"):
        try:
            # Create dummy variables and interaction terms for the prediction
            input_data = pd.DataFrame({
                "Mean Temp (°C)": [mean_temp],
                "Total Precip (mm)": [total_precip],
                "Total Snow (cm)": [total_snow],
                "Cool Deg Days (°C)": [cool_deg_days]
            })

            # Add dummy variables for day_of_week
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
                input_data[f'day_of_week_{day}'] = 1 if day == day_of_week else 0

            # Add interaction terms
            for day_col in [col for col in input_data.columns if 'day_of_week' in col]:
                input_data[f'MeanTemp_{day_col}'] = input_data['Mean Temp (°C)'] * input_data[day_col]

            # Align columns with training data
            feature_order = [
                "Mean Temp (°C)", "Total Precip (mm)", "Total Snow (cm)", "Cool Deg Days (°C)",
                "day_of_week_Tuesday", "day_of_week_Wednesday", "day_of_week_Thursday",
                "day_of_week_Friday", "day_of_week_Saturday", "day_of_week_Sunday",
                "MeanTemp_day_of_week_Tuesday", "MeanTemp_day_of_week_Wednesday",
                "MeanTemp_day_of_week_Thursday", "MeanTemp_day_of_week_Friday",
                "MeanTemp_day_of_week_Saturday", "MeanTemp_day_of_week_Sunday"
            ]

            input_data = input_data.reindex(columns=feature_order, fill_value=0)

            model_path = f"models/{selected_bikelane}_gam_model.pkl"
            with open(model_path, "rb") as model_file:
                gam_model = joblib.load(model_file)

            predicted_usage = max(0, gam_model.predict(input_data)[0])

            st.success(f"Predicted Bike Usage: {predicted_usage:.2f}")

            # Add prediction to visualizations
            prediction_point = pd.DataFrame({
                "Total Precip (mm)": [total_precip],
                "Mean Temp (°C)": [mean_temp],
                "num_usage": [predicted_usage],
                "bikelane": [selected_bikelane],
                "type": ["Prediction"]
            })

            filtered_updated_data = pd.concat([
                bike_weather_melt[bike_weather_melt['bikelane'] == selected_bikelane].assign(type="Historical"),
                prediction_point
            ])

            scatter_tp_bl_updated = alt.Chart(filtered_updated_data).mark_point().encode(
                x="Total Precip (mm):Q",
                y="num_usage:Q",
                color=alt.condition(alt.datum.type == 'Prediction', alt.value('red'), alt.value('#87CEEB')),
                tooltip=["Total Precip (mm)", "num_usage", "type"]
            ).properties(
                width=400,
                height=400
            )

            scatter_mt_bl_updated = alt.Chart(filtered_updated_data).mark_point().encode(
                x="Mean Temp (°C):Q",
                y="num_usage:Q",
                color=alt.condition(alt.datum.type == 'Prediction', alt.value('red'), alt.value('#87CEEB')),
                tooltip=["Mean Temp (°C)", "num_usage", "type"]
            ).properties(
                width=400,
                height=400
            )

            st.altair_chart(scatter_tp_bl_updated & scatter_mt_bl_updated)

        except Exception as e:
            st.error(f"Error in prediction or visualization: {e}")

with col2:
    st.subheader("Summary View and Visualizations")

    # Model Performance (Actual vs Predicted)
    try:
        # Load pre-saved X_test and y_test for the selected bike lane
        X_test_path = f"models/{selected_bikelane}_X_test.pkl"
        y_test_path = f"models/{selected_bikelane}_y_test.pkl"

        X_test = pd.read_pickle(X_test_path)
        y_test = pd.read_pickle(y_test_path)

        # Load the model
        model_path = f"models/{selected_bikelane}_gam_model.pkl"
        with open(model_path, "rb") as model_file:
            gam_model = joblib.load(model_file)

        # Predict using the loaded model
        y_predicted = gam_model.predict(X_test.to_numpy())

        # Create Actual vs Predicted DataFrame
        actual_vs_predicted = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_predicted
        })

        # Create scatter plot for Actual vs Predicted
        scatter_actual_vs_predicted = alt.Chart(actual_vs_predicted).mark_point().encode(
            x="Actual:Q",
            y="Predicted:Q",
            tooltip=["Actual", "Predicted"]
        )

        # Create line of best fit (identity line)
        line_of_best_fit = alt.Chart(pd.DataFrame({
            "Actual": [actual_vs_predicted["Actual"].min(), actual_vs_predicted["Actual"].max()],
            "Predicted": [actual_vs_predicted["Actual"].min(), actual_vs_predicted["Actual"].max()]
        })).mark_line(color="red", strokeDash=[5, 5]).encode(
            x="Actual:Q",
            y="Predicted:Q"
        )

        combined_actual_vs_predicted = (scatter_actual_vs_predicted + line_of_best_fit).properties(
            width=1000,
            height=500,
            title="Actual vs Predicted with Line of Best Fit"
        )

        st.altair_chart(combined_actual_vs_predicted)

    except Exception as e:
        st.error(f"Error in Model Performance visualization: {e}")

    # Additional Visualizations
    selection = alt.selection(type="multi", fields=["year/month"])

    bike_usage_by_month = alt.Chart(bike_weather_melt).mark_bar().encode(
        y="num_usage",
        x="year/month",
        color=alt.condition(selection, alt.value("orange"), alt.value("lightgrey"))
    ).add_selection(selection).properties(height=250, width=500)

    usage_per_bike_lane = alt.Chart(bike_weather_melt).mark_bar().encode(
        y="num_usage",
        x="bikelane"
    ).transform_filter(selection).properties(height=250, width=500)

    st.altair_chart(bike_usage_by_month | usage_per_bike_lane)

    min_temp_chart = alt.Chart(bike_weather_melt).mark_line().encode(
        y="Min Temp (°C)",
        x="date",
        color=alt.condition(selection, alt.value("orange"), alt.value("lightgrey"))
    ).add_selection(selection).properties(height=250, width=500)

    max_temp_chart = alt.Chart(bike_weather_melt).mark_line().encode(
        y="Max Temp (°C)",
        x="date",
        color=alt.condition(selection, alt.value("orange"), alt.value("lightgrey"))
    ).add_selection(selection).properties(height=250, width=500)

    st.altair_chart(min_temp_chart | max_temp_chart)

    total_precipt_chart = alt.Chart(bike_weather_melt).mark_line().encode(
        y="Total Precip (mm)",
        x="date",
        color=alt.condition(selection, alt.value("orange"), alt.value("lightgrey"))
    ).add_selection(selection).properties(height=250, width=1000)

    st.altair_chart(total_precipt_chart)
