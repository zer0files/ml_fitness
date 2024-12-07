import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Custom algorithm for counting repetitions (example logic)
def count_repetitions(data, threshold=0.5):
    repetitions = 0
    for i in range(1, len(data)):
        if abs(data[i] - data[i-1]) > threshold:
            repetitions += 1
    return repetitions

# Function to process uploaded data and predict repetitions
def process_and_predict(file, threshold):
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Check if the file has the necessary columns
    if 'sensor_value' not in df.columns:
        st.error("Error: 'sensor_value' column not found in the uploaded data.")
        return None
    
    # Count repetitions using the custom algorithm
    sensor_values = df['sensor_value'].values
    repetitions = count_repetitions(sensor_values, threshold)
    
    # Plotting the sensor data for visualization
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], sensor_values)  # Assuming there's a 'timestamp' column
    plt.xlabel('Time')
    plt.ylabel('Sensor Value')
    plt.title('Sensor Value Over Time')
    
    # Display the plot
    st.pyplot(plt)

    # Return the result
    return repetitions

# Streamlit interface
def streamlit_interface():
    st.title("Sensor Data Repetition Detector")
    st.markdown("Upload your CSV file containing time-series sensor data to detect repetitions.")

    # File upload input
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    # Slider to control threshold for counting repetitions
    threshold = st.slider("Threshold", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    # Button to trigger prediction
    if uploaded_file is not None:
        if st.button("Detect Repetitions"):
            # Process the file and predict repetitions
            repetitions = process_and_predict(uploaded_file, threshold)
            if repetitions is not None:
                st.success(f"Detected repetitions: {repetitions}")

# Run the Streamlit interface
if __name__ == "__main__":
    streamlit_interface()
