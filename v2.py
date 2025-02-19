import streamlit as st
import pandas as pd
import plotly.express as px

# Function to load input and output CSVs
def load_data(input_csv_path, output_csv_path):
    input_data = pd.read_csv(input_csv_path, sep='\t')
    output_data = pd.read_csv(output_csv_path, sep='\t')
    return input_data, output_data

# Function to display input CSV
def display_input_data(input_data):
    st.subheader("📂 Input Dataset")
    st.dataframe(input_data)  # Improved table display
    st.info(f"📌 **Total rows in Input Dataset:** {len(input_data)}")

# Function to display output CSV
def display_output_data(output_data):
    st.subheader("📂 Output Dataset")
    st.dataframe(output_data)
    st.info(f"📌 **Total rows in Output Dataset:** {len(output_data)}")

# Function to display missing rows
def display_missing_rows(input_data, output_data):
    missing_rows = input_data[~input_data.isin(output_data)].dropna()
    missing_count = len(missing_rows)

    st.subheader("❌ Rows Present in Input CSV But Not in Output CSV")
    
    if missing_count > 0:
        st.dataframe(missing_rows)  # Show missing rows in a better format
    else:
        st.success("✅ No missing rows. All rows from input are present in output.")

    # Summary
    st.info(f"📌 **Total rows in Input Dataset:** {len(input_data)}")
    st.info(f"📌 **Total rows in Output Dataset:** {len(output_data)}")

    if missing_count > 0:
        st.error(f"❌ **Total missing rows:** {missing_count}")
    else:
        st.success("✅ No missing rows.")

# Function to display comparison graph with summary
def display_comparison_graph(input_data, output_data):
    missing_count = len(input_data) - len(output_data)  

    df_plot = pd.DataFrame({
        "Dataset": ["Input Dataset", "Output Dataset"],
        "Rows": [len(input_data), len(output_data)]
    })

    fig = px.bar(df_plot, x="Dataset", y="Rows", text_auto=True, color="Dataset",
                 title="📊 Comparison of Number of Rows",
                 color_discrete_map={"Input Dataset": "#1f77b4", "Output Dataset": "#ff7f0e"})  

    fig.update_traces(marker=dict(line=dict(color='black', width=1)))  
    fig.update_layout(yaxis_title="Number of Rows")

    st.plotly_chart(fig)

    # Summary section below the graph
    st.subheader("📊 Summary")
    st.info(f"📌 **Total rows in Input Dataset:** {len(input_data)}")
    st.info(f"📌 **Total rows in Output Dataset:** {len(output_data)}")

    if missing_count > 0:
        st.error(f"❌ **Total missing rows:** {missing_count}")
    else:
        st.success("✅ No missing rows.")

# Main function
def main():
    st.set_page_config(page_title="Feedback Validation", layout="wide")
    st.title("🔍 Authentic Feedback Validation Framework")

    # Specify file paths
    input_csv_path = "dataset/review.csv"
    output_csv_path = "./real_reviews.csv"

    # Load CSVs
    input_data, output_data = load_data(input_csv_path, output_csv_path)

    # Sidebar Navigation
    with st.sidebar:
        st.header("🔎 Navigation")
        selected_option = st.radio("Choose a section:", 
                                   ["📂 Input Dataset", "📂 Output Dataset", "❌ Missing Rows", "📊 Comparison Graph"])

    # Display sections based on selection
    if selected_option == "📂 Input Dataset":
        display_input_data(input_data)
    elif selected_option == "📂 Output Dataset":
        display_output_data(output_data)
    elif selected_option == "❌ Missing Rows":
        display_missing_rows(input_data, output_data)
    elif selected_option == "📊 Comparison Graph":
        display_comparison_graph(input_data, output_data)

if __name__ == "__main__":
    main()
