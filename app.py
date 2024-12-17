import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the Beijing Air Quality dataset
@st.cache_data
def load_data():
    # Replace with the correct path to your dataset
    return pd.read_csv("https://raw.githubusercontent.com/hy5373/hyaiq/refs/heads/main/preprocessed_air_quality_data.csv")

# Load the saved Random Forest model
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model_compressed.pkl")
    station_encoder = joblib.load("station_encoder.pkl")
    return model, station_encoder

# Dataset Description
def describe_dataset():
    st.write("""
    ### About the Dataset
    This dataset contains air quality measurements collected from various stations in Beijing.
    It includes hourly readings of key air pollutants and meteorological factors. The main goal is to predict PM2.5 levels,
    which represent fine particulate matter in the air and are a critical indicator of air quality.

    #### Column Descriptions:
    - **year, month, day**: Date components of the measurement.
    - **PM2.5**: Fine particulate matter concentration (µg/m³).
    - **PM10**: Particulate matter concentration (µg/m³).
    - **SO2**: Sulfur dioxide concentration (µg/m³).
    - **NO2**: Nitrogen dioxide concentration (µg/m³).
    - **CO**: Carbon monoxide concentration (mg/m³).
    - **O3**: Ozone concentration (µg/m³).
    - **DEWP**: Dew point temperature (°C).
    - **station**: The monitoring station where the data was collected.
    """)

# Define the main function
def main():
    st.title("Air Quality Analysis and Prediction App")

    # Sidebar for navigation
    menu = ["Data Overview", "EDA", "Modeling and Prediction"]
    choice = st.sidebar.selectbox("Navigate", menu)

    # Load dataset and model
    df = load_data()
    model, station_encoder = load_model()

    # Data Overview Section
    if choice == "Data Overview":
        st.header("Data Overview")
        describe_dataset()
        st.write("This section provides a preview and summary of the dataset.")
        st.dataframe(df.head(5))  # Display the first few rows
        st.write("Dataset Shape:", df.shape)
        st.write("Statistical Summary:")
        st.write(df.describe().T)
        st.write("Unique Stations in the dataset")
        st.write(df['station'].unique().tolist())

    # EDA Section
    elif choice == "EDA":
        st.header("Exploratory Data Analysis (EDA)")
        st.write("This section includes visualizations and insights from the dataset.")

        # Missing values
        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.write(missing_values[missing_values > 0])
        else:
            st.success("No missing values found in the dataset!")

        # Numeric columns for correlation
        numeric_columns = df.select_dtypes(include=["number"]).columns

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        if not numeric_columns.empty:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df[numeric_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm")
            st.pyplot(plt)
            plt.close()

        # Distribution of PM2.5
        st.subheader("PM2.5 Distribution")
        plt.figure(figsize=(8, 5))
        sns.histplot(df["PM2.5"], kde=True, bins=30, color="blue")
        st.pyplot(plt)
        plt.close()

        # Average Pollutant Concentrations
        st.subheader("Average Pollutant Concentrations")
        pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        colors = ["red", "blue", "green", "orange", "purple", "brown"]
        averages = df[pollutants].mean()
        plt.figure(figsize=(10, 6))
        plt.bar(pollutants, averages, color=colors)
        plt.xlabel("Pollutant")
        plt.ylabel("Average Concentration")
        plt.title("Average Pollutant Concentrations")
        st.pyplot(plt)
        plt.close()

    # Modeling and Prediction Section
    elif choice == "Modeling and Prediction":
        st.header("Modeling and Prediction")
        st.write("Provide input values to predict PM2.5 levels.")

        # Input fields for user data
        year = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2025, step=1)
        month = st.number_input("Enter Month (1-12)", min_value=1, max_value=12, value=12, step=1)
        day = st.number_input("Enter Day (1-31)", min_value=1, max_value=31, value=31, step=1)
        pm10 = st.number_input("Enter PM10 Value(0-500)", min_value=0.0, max_value=500.0, value=33.0, step=1.0)
        so2 = st.number_input("Enter SO2 Value(0-500)", min_value=0.0, max_value=500.0, value=33.0, step=1.0)
        no2 = st.number_input("Enter NO2 Value(0-500)", min_value=0.0, max_value=500.0, value=33.0, step=1.0)
        co = st.number_input("Enter CO Value(0-1000)", min_value=0.0, max_value=1000.0, value=33.0, step=1.0)
        o3 = st.number_input("Enter O3 Value(0-500)", min_value=0.0, max_value=500.0, value=33.0, step=1.0)
        dewp = st.number_input("Enter DEWP Value(-50-30)", min_value=-50.0, max_value=30.0, value=20.0, step=1.0)
        station_name = st.text_input("Enter Station Name (e.g., 'Guanyuan')")

        # When the predict button is clicked
        if st.button("Predict PM2.5"):
            valid_stations = station_encoder.classes_.tolist()
            if station_name not in valid_stations:
                st.error(f"Invalid station name. Valid stations are: {', '.join(valid_stations)}")
            else:
                try:
                    # Encode station name
                    station_encoded = station_encoder.transform([station_name])[0]

                    # Create a DataFrame for prediction
                    input_data = pd.DataFrame([{
                        "year": year,
                        "month": month,
                        "day": day,
                        "PM10": pm10,
                        "SO2": so2,
                        "NO2": no2,
                        "CO": co,
                        "O3": o3,
                        "DEWP": dewp,
                        "station": station_encoded
                    }])

                    # Predict PM2.5
                    prediction = model.predict(input_data)[0]

                    # Display the prediction
                    st.success(f"Predicted PM2.5 Value: {prediction:.2f}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Run the app
if __name__ == "__main__":
    main()
