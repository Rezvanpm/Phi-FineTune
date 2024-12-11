import os
import streamlit as st
import pandas as pd
from preprocessing import preprocess_data
from LMs import load_language_model, process_pipeline

# تنظیمات Streamlit
os.environ["STREAMLIT_BROWSER_GATHERUSAGESTATS"] = "false"
st.title("Language Model and Techniques Pipeline")

st.title("پلتفرم پردازش داده و مدل‌سازی زبانی")

# Step 1: انتخاب دیتاست
st.write("### Step 1: Select Dataset")
datasets = [file for file in os.listdir('./datasets') if file.endswith(".csv")]
selected_dataset = st.selectbox("Select a dataset:", datasets)


# Step 2: انتخاب تکنیک‌های پیش‌پردازش
st.write("### Step 2: Select Preprocessing Techniques")
preprocessing_options = st.multiselect(
    "Choose preprocessing techniques:",
    [
        "Remove Missing Values",
        "Convert Categorical to Numeric",
        "Normalize Column",
        "Remove Noise",
        "Remove Irrelevant Columns",
        "Convert to Datetime",
        "Standardize Column",
        "Feature Engineering"
    ]
)

# Step 3: انتخاب مدل زبانی 
st.write("### Step 3: Select LMs")
methods = ["RAG", "Fine-Tuning", "RAFT"]
selected_method = st.radio("Choose a language model:", methods)


# Step 4: انتخاب تکنیک
st.write("### Step 4: Select Method")
methods = ["RAG", "Fine-Tuning", "RAFT"]
selected_method = st.radio("Choose a method:", methods)

# Step 5: بارگذاری مدل زبانی و اجرای پایپ‌لاین
if st.button("Start Process"):
    try:
        # بارگذاری دیتاست
        file_path = os.path.join('./datasets', selected_dataset)
        data = pd.read_csv(file_path)

        # بارگذاری مدل زبانی
        st.info("Loading language model...")
        model_client = load_language_model()

        # اجرای پایپ‌لاین
        with st.spinner("Processing..."):
            pipeline_results = process_pipeline(
                data, preprocessing_options, selected_method, model_client)
            st.success("Pipeline completed!")
            st.json(pipeline_results)

    except Exception as e:
        st.error(f"An error occurred: {e}")
