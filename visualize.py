import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load input and output CSVs
def load_data(input_csv_path, output_csv_path):
    input_data = pd.read_csv(input_csv_path, sep='\t')
    output_data = pd.read_csv(output_csv_path, sep='\t')
    return input_data, output_data

# Function to display input CSV
def display_input_data(input_data):
    num_rows_input, num_cols_input = input_data.shape
    st.subheader("Input Dataset")
    st.write(input_data)
    st.write(f"Number of rows: {num_rows_input}")
    st.write(f"Number of columns: {num_cols_input}")

# Function to display output CSV
def display_output_data(output_data):
    num_rows_output, num_cols_output = output_data.shape
    st.subheader("Output Dataset")
    st.write(output_data)
    st.write(f"Number of rows: {num_rows_output}")
    st.write(f"Number of columns: {num_cols_output}")

# Function to display rows present in input CSV but not in output CSV
def display_missing_rows(input_data, output_data):
    input_not_in_output = input_data[~input_data.isin(output_data)].dropna()
    if not input_not_in_output.empty:
        st.subheader("Rows Present in Input CSV But Not in Output CSV")
        st.write(input_not_in_output)
    else:
        st.write("All rows from input CSV are present in output CSV.")

# Function to display a graph or block diagram comparing input and output CSVs
def display_comparison_graph(input_data, output_data):
    fig, ax = plt.subplots()
    ax.plot(["Input CSV", "Output CSV"], [input_data.shape[0], output_data.shape[0]], marker='o', linestyle='-')
    ax.set_ylabel("Number of Rows")
    ax.set_title("Comparison of Number of Rows Between Input and Output CSVs")
    st.pyplot(fig)

def main():
    st.title("Authentic Feedback Validation Framework")

    # Specify file paths
    input_csv_path = "dataset/review.csv"
    output_csv_path = "./real_reviews.csv"

    # Load input and output CSVs
    input_data, output_data = load_data(input_csv_path, output_csv_path)

    # Create horizontal buttons for selecting dataset
    cols = st.columns(4)
    if cols[0].button("Input Dataset"):
        display_input_data(input_data)
    if cols[1].button("Output Dataset"):
        display_output_data(output_data)
    if cols[2].button("Missing Rows"):
        display_missing_rows(input_data, output_data)
    if cols[3].button("Comparison Graph"):
        display_comparison_graph(input_data, output_data)

if __name__ == "__main__":
    main()
