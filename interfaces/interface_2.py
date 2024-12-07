
### Enhanced Gradio Interface with File Upload, Data Visualization, and Custom Algorithm

import gradio as gr
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
    df = pd.read_csv(file.name)
    
    # Check if the file has the necessary columns
    if 'sensor_value' not in df.columns:
        return "Error: 'sensor_value' column not found in the uploaded data.", None
    
    # Count repetitions using the custom algorithm
    sensor_values = df['sensor_value'].values
    repetitions = count_repetitions(sensor_values, threshold)
    
    # Plotting the sensor data for visualization
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], sensor_values)  # Assuming there's a 'timestamp' column
    plt.xlabel('Time')
    plt.ylabel('Sensor Value')
    plt.title('Sensor Value Over Time')
    
    # Save the plot to a file
    plot_file = "sensor_plot.png"
    plt.savefig(plot_file)
    plt.close()

    # Return the result and the plot file
    return f"Detected repetitions: {repetitions}", plot_file

# Gradio interface
def gradio_interface():
    with gr.Blocks() as iface:
        gr.Markdown("# Sensor Data Repetition Detector")
        gr.Markdown("Upload your CSV file containing time-series sensor data to detect repetitions.")

        # File upload input
        file_input = gr.File(label="Upload CSV file")

        # Slider to control threshold for counting repetitions
        threshold_slider = gr.Slider(label="Threshold", minimum=0.1, maximum=2.0, value=0.5, step=0.1)

        # Button to trigger prediction
        submit_btn = gr.Button("Detect Repetitions")

        # Text output for displaying detected repetitions
        text_output = gr.Textbox(label="Detected Repetitions")

        # Image output for displaying plot
        plot_output = gr.Image(label="Sensor Data Plot")

        # Action: When the button is clicked, call the `process_and_predict` function
        submit_btn.click(
            fn=process_and_predict,
            inputs=[file_input, threshold_slider],
            outputs=[text_output, plot_output]
        )

    iface.launch()

# Run the Gradio interface
if __name__ == "__main__":
    gradio_interface()


### Key Features:
# 1. *File Upload*: Allows the user to upload a CSV file containing time-series data.
# 2. *Threshold Adjustment*: A slider enables the user to adjust the threshold for counting repetitions.
# 3. *Visualization*: The code generates and displays a time-series plot of the sensor data.
# 4. *Error Handling*: Checks if the necessary 'sensor_value' column exists in the uploaded CSV file.
# 5. *Outputs*: Displays the detected repetitions as text and provides a plot of the sensor data.

# ### How It Works:
# - *File Input*: The user uploads a CSV file containing time-series data.
# - *Prediction*: The process_and_predict function reads the file, counts repetitions based on a threshold, and generates a plot.
# - *Outputs*: Displays the count of repetitions and a visualization of the sensor data over time.

# This interface is easy to use and provides more control over the prediction and visualization of the data. Let me know if you'd like further customizations or improvements!