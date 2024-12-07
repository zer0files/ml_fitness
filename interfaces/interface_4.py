import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from math import log2
from scipy.stats import chisquare

st.set_page_config(page_title='InterFace cry 4', page_icon='🧩')


def main():
    st.title("📟 Sensor Data Repetition Detector")
    st.write("Upload your CSV file containing time-series sensor data to detect repetitions.")
    uploaded_file_1 = st.file_uploader("",type=["csv"],key=1)
    uploaded_file_2 = st.file_uploader("",type=["csv"],key=2)

if __name__ == "__main__":
    main()