import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


from glob import glob
import numpy as np
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
import pickle
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

import os
# from file_sel import select_files_separately;

exercise_class = ""


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]= (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

fs = 1000 / 200
LowPass = LowPassFilter()
# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency = fs, cutoff_frequency=cutoff, order=order
    )
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]
    
    # PLOTING -----------------------
    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"],"o",color="red")
    ax.set_ylabel(f"{column}_lowpass")
    # exercise = dataset["label"].iloc[0].title()
    exercise = exercise_class
    # category = dataset["category"].iloc[0].title()
    plt.title(f"{exercise}: {len(peaks)} Reps")
    # plt.show()
    
    # Display the plot
    st.pyplot(plt)

    return len(peaks)

def count_reps_label(dataset,label):
    
    df=dataset.copy()
    
    acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
    gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
    df["acc_r"] = np.sqrt(acc_r)
    df["gyr_r"] = np.sqrt(gyr_r)
    
    
    column = "acc_r"
    cutoff = 0.4
    if label == "squat":
        cut_off = 0.35
    
    if label == "row":
        cut_off = 0.65
        col = "gyr_x"
        
    if label == "ohp":
        cut_off = 0.35
    
    reps = count_reps(df, cutoff=cutoff, column=column)
    print(reps)
    return reps




# Custom algorithm for counting repetitions (example logic)
def count_repetitions(data, threshold=0.5):
    repetitions = 0
    for i in range(1, len(data)):
        if abs(data[i] - data[i-1]) > threshold:
            repetitions += 1
    return repetitions

# Function to process uploaded data and predict repetitions
def process_and_predict(file_1,file_2):
    # Read the CSV file
    # df = pd.read_csv(file1)
    acc_df = pd.read_csv(file_1)
    gyr_df = pd.read_csv(file_2)
    
    
    # --------------------------------------------------------------
    # Working with datetimes
    # --------------------------------------------------------------

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    # --------------------------------------------------------------
    # Merging datasets
    # --------------------------------------------------------------

    data_merged =  pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)

    data_merged.columns =[
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
    ]
    
    # --------------------------------------------------------------
    # Resample data (frequency conversion)
    # --------------------------------------------------------------

    # Accelerometer:    12.500HZ
    # Gyroscope:        25.000Hz


    sampling ={
        "acc_x": "mean",
        "acc_y": "mean",
        "acc_z": "mean",
        "gyr_x": "mean",
        "gyr_y": "mean",
        "gyr_z": "mean",
    }

    data_merged.resample(rule="S").apply(sampling)


    days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
    data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])
    data_resampled.info()

    # OUTPUT : data_resampled
    
    # # 3
    # # build_features.py

    df = data_resampled

    predictor_columns = list(df.columns)

    # --------------------------------------------------------------
    # Dealing with missing values (imputation)
    # --------------------------------------------------------------
    for col in predictor_columns:
        df[col] = df[col].interpolate()

    df.info()
    
    # --------------------------------------------------------------
    # Butterworth lowpass filter
    # --------------------------------------------------------------

    df_lowpass = df.copy()
    LowPass = LowPassFilter()

    fs = 1000 / 200
    cutoff = 1.3

    df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

    for col in predictor_columns:
        df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
        df_lowpass[col] = df_lowpass[col + "_lowpass"]
        del df_lowpass[col + "_lowpass"]
        
    # --------------------------------------------------------------
    # Principal component analysis PCA
    # --------------------------------------------------------------

    df_pca = df_lowpass.copy()
    PCA = PrincipalComponentAnalysis()

    pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

    df_pca  =PCA.apply_pca(df_pca, predictor_columns, 3)

    # --------------------------------------------------------------
    # Sum of squares attributes
    # --------------------------------------------------------------
    df_squared = df_pca.copy()

    acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
    gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

    df_squared["acc_r"] = np.sqrt(acc_r)
    df_squared["gyr_r"] = np.sqrt(gyr_r)

    # --------------------------------------------------------------
    # Temporal abstraction
    # --------------------------------------------------------------

    df_temporal = df_squared.copy()
    NumAbs = NumericalAbstraction()

    predictor_columns = predictor_columns + ["acc_r", "gyr_r"] # <------CHECK-----part5b 10:00min-----gyr_y->gyr_r--

    ws = int(1000 / 200)

    for col in predictor_columns:
        df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
        df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

    # --------------------------------------------------------------
    # Dealing with overlapping windows
    # --------------------------------------------------------------

    df_freq = df_temporal.dropna()
    # df_freq = df_freq.iloc[::2]

    # --------------------------------------------------------------
    # Clustering
    # --------------------------------------------------------------

    df_cluster = df_freq.copy()
    
    
    # #***************************************************LOAD Random Forest model (train model)

    df = df_cluster
    df_col = list(df.columns)


    ## my mod


    # --------------------------------------------------------------
    # Split feature subsets
    # --------------------------------------------------------------

    basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    square_features = ["acc_r", "gyr_r"]
    pca_features = ["pca_1", "pca_2", "pca_3"]
    time_features = [f for f in df.columns if "_temp_" in f]
    freq_features = [f for f in df.columns if ("_freq" in f) or ("_pse" in f)]

    feature_set_1 = list(set(basic_features))
    feature_set_2 = list(set(basic_features + square_features + pca_features))
    feature_set_3 = list(set(feature_set_2 + time_features))
    # feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))
    feature_set_4 = list(set(feature_set_3 + freq_features))
    
    # PREDICT

    # Load the saved model
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, 'random_forest_model.pkl')

    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
        
    single_data_reshaped = df.iloc[:1].copy()
    # single_data_reshaped_feature_set_4_col = list(single_data_reshaped[feature_set_4].columns)

    single_prediction = loaded_model.predict(single_data_reshaped[sorted(feature_set_3)])

    print("Predicted class label for the single data point:", single_prediction)
    global exercise_class
    exercise_class = single_prediction[0]




    ### NIKHIL CODE END ###
    
    
    # Check if the file has the necessary columns
    # if 'sensor_value' not in df.columns:
    #     st.error("Error: 'sensor_value' column not found in the uploaded data.")
    #     return None
    
    # Count repetitions using the custom algorithm
    # sensor_values = df['sensor_value'].values
    repetitions = count_reps_label(data_resampled, exercise_class)
    
    # Plotting the sensor data for visualization
    # plt.figure(figsize=(10, 5))
    # plt.plot(df['timestamp'], sensor_values)  # Assuming there's a 'timestamp' column
    # plt.xlabel('Time')
    # plt.ylabel('Sensor Value')
    # plt.title('Sensor Value Over Time')
    
    # # Display the plot
    # st.pyplot(plt)

    # Return the result
    return repetitions

# Streamlit interface
def streamlit_interface():
    st.title("📟 Sensor Data Repetition Detector")
    st.markdown("Upload your CSV file containing time-series sensor data to detect repetitions.")

    # File upload input (Accelerometer)
    uploaded_file_1 = st.file_uploader("Upload Accelerometer CSV file", type="csv",key=1)
    
    # File upload input (Gyroscope)
    uploaded_file_2 = st.file_uploader("Upload Gyroscope CSV file", type="csv",key=2)
    

    # Slider to control threshold for counting repetitions
    # threshold = st.slider("Threshold", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    # Button to trigger prediction
    if (uploaded_file_1 is not None)and(uploaded_file_2 is not None):
        if st.button("Detect Repetitions"):
            # Process the file and predict repetitions
            repetitions = process_and_predict(uploaded_file_1, uploaded_file_2)
            if repetitions is not None:
                st.success(f"Detected repetitions: {repetitions}")

# Run the Streamlit interface
if __name__ == "__main__":
    streamlit_interface()