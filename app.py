import streamlit as st
import pandas as pd
import altair as alt
import joblib
import sys
import os

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

# Layout adjustments
st.header("Filters and Predictions")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Filter Options")
    selected_bikelane = st.selectbox(
        "Select a Bike Lane:",
        options=bike_weather_melt['bikelane'].unique(),
        index=0
    )

    st.subheader("Prediction Inputs")
    mean_temp = st.number_input("Mean Temperature (°C):", value=15.0)
    total_precip = st.number_input("Total Precipitation (mm):", value=0.0)
    total_snow = st.number_input("Total Snow (cm):", value=0.0)
    day_of_week = st.selectbox("Day of Week:", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=0)

    cool_deg_days = mean_temp - 18  # Calculate Cool Degree Days

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
            feature_order = ["Mean Temp (°C)", "Total Precip (mm)", "Total Snow (cm)", "Cool Deg Days (°C)",
                             "day_of_week_Monday", "day_of_week_Saturday", "day_of_week_Sunday",
                             "day_of_week_Thursday", "day_of_week_Tuesday", "day_of_week_Wednesday",
                             "MeanTemp_day_of_week_Monday", "MeanTemp_day_of_week_Saturday",
                             "MeanTemp_day_of_week_Sunday", "MeanTemp_day_of_week_Thursday",
                             "MeanTemp_day_of_week_Tuesday", "MeanTemp_day_of_week_Wednesday"]

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
            st.error(f"Error in prediction: {e}")

with col2:
    # Section 1: Summary Visualization
    st.subheader("Summary View")
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

    # Section 2: Filtered Visualization
    st.subheader(f"Visualizations for {selected_bikelane}")
    filtered_data = bike_weather_melt[bike_weather_melt['bikelane'] == selected_bikelane]

    filtered_line_chart = alt.Chart(filtered_data).mark_line().encode(
        x="year/month:T",
        y="mean(num_usage):Q",
        color="bikelane:N"
    ).properties(
        width=1000,
        height=250
    )

    st.altair_chart(filtered_line_chart)

    # Linked scatterplots
    scatter_base = alt.Chart(filtered_data).properties(width=400, height=400)

    scatter_tp_bl = scatter_base.mark_circle().encode(
        x="Total Precip (mm)",
        y="num_usage",
        color="bikelane:N",
        tooltip=["Total Precip (mm)", "num_usage"]
    )

    scatter_mt_bl = scatter_base.mark_circle().encode(
        x="Mean Temp (°C)",
        y="num_usage",
        color="bikelane:N",
        tooltip=["Mean Temp (°C)", "num_usage"]
    )

    st.altair_chart(scatter_tp_bl | scatter_mt_bl)
