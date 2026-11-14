
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys


# 1. PROJECT PATHS

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

DATA_DIR = PROJECT_DIR / "Data"

sys.path.insert(0, str(APP_DIR))

from preprocessing import prepare_input_data
from prediction import (
    load_models,
    predict_sales_and_customers
)


# 2. PAGE CONFIGURATION

st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide"
)


# 3. LOAD MODELS AND STORE DATA

@st.cache_resource
def get_models():
    return load_models()


@st.cache_data
def get_store_data():
    return pd.read_csv(DATA_DIR / "store.csv")


try:
    sales_model, customers_model = get_models()
    store_df = get_store_data()

except Exception as e:
    st.error(f"Unable to load models or store data: {e}")
    st.stop()


# 4. APPLICATION HEADER

st.title("📊 Rossmann Store Sales Forecasting")

st.markdown(
    """
    ### Sales and Customer Prediction System

    Forecast daily sales and customer numbers for Rossmann stores
    using trained Machine Learning models.

    Select a store, upload your forecasting data,
    and generate predictions for up to six weeks.
    """
)

st.divider()


# 5. STORE SELECTION

st.subheader("1. Select Store")

store_ids = sorted(
    store_df["Store"].dropna().unique()
)

selected_store = st.selectbox(
    "Choose Store ID",
    store_ids
)

st.write(f"Selected Store: **{selected_store}**")


# 6. CSV UPLOAD

st.subheader("2. Upload Forecasting Data")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

st.caption(
    "Required columns: Date, Open, Promo, "
    "StateHoliday and SchoolHoliday. "
    "If Store is not provided, the selected Store ID is used."
)


# 7. PROCESS UPLOADED DATA

if uploaded_file is not None:

    try:

        raw_df = pd.read_csv(
            uploaded_file,
            dtype={"StateHoliday": "string"}
        )

        st.subheader("Uploaded Data Preview")

        st.dataframe(
            raw_df.head(),
            use_container_width=True
        )

        # Validate Date
        if "Date" not in raw_df.columns:
            st.error("The CSV must contain a Date column.")
            st.stop()

        raw_df["Date"] = pd.to_datetime(
            raw_df["Date"],
            errors="coerce"
        )

        if raw_df["Date"].isna().any():
            st.error("Invalid or missing dates found.")
            st.stop()

        # Use selected Store ID
        if "Store" in raw_df.columns:

            raw_df["Store"] = pd.to_numeric(
                raw_df["Store"],
                errors="coerce"
            )

            raw_df = raw_df[
                raw_df["Store"] == selected_store
            ].copy()

        else:

            raw_df["Store"] = selected_store

        if raw_df.empty:
            st.warning(
                "No records found for the selected store."
            )
            st.stop()

        # Check duplicate dates
        if raw_df["Date"].duplicated().any():
            st.error(
                "Duplicate dates found for this store."
            )
            st.stop()

        # Check forecasting period
        forecast_days = (
            raw_df["Date"].max()
            - raw_df["Date"].min()
        ).days + 1

        if forecast_days > 42:
            st.error(
                "Please upload data covering no more than "
                "six weeks (42 calendar days)."
            )
            st.stop()

        st.success(
            f"{len(raw_df)} records found for Store "
            f"{selected_store}."
        )

        st.write(
            f"Forecast period: {forecast_days} calendar days"
        )

        # 8. GENERATE PREDICTIONS
    

        if st.button(
            "Generate Sales and Customer Predictions",
            type="primary"
        ):

            with st.spinner(
                "Preparing data and generating predictions..."
            ):

                # Feature engineering
                processed_df = prepare_input_data(
                    raw_df,
                    store_df
                )

                # Generate predictions
                result_df = predict_sales_and_customers(
                    processed_df,
                    sales_model,
                    customers_model
                )

                result_df = result_df.sort_values(
                    "Date"
                ).reset_index(drop=True)

            st.success(
                "Sales and customer predictions generated successfully!"
            )

            st.divider()


            # 9. PREDICTION SUMMARY
    

            st.subheader("3. Prediction Summary")

            total_sales = result_df[
                "Predicted_Sales"
            ].sum()

            total_customers = result_df[
                "Predicted_Customers"
            ].sum()

            average_sales = result_df[
                "Predicted_Sales"
            ].mean()

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Predicted Sales",
                f"{total_sales:,.2f}"
            )

            col2.metric(
                "Total Predicted Customers",
                f"{total_customers:,.0f}"
            )

            col3.metric(
                "Average Daily Sales",
                f"{average_sales:,.2f}"
            )

            st.divider()

    
            # 10. SALES FORECAST CHART
    

            st.subheader("4. Daily Sales Forecast")

            fig_sales = px.line(
                result_df,
                x="Date",
                y="Predicted_Sales",
                markers=True,
                title="Predicted Daily Sales",
                template="plotly_white"
            )

            fig_sales.update_layout(
                xaxis_title="Date",
                yaxis_title="Predicted Sales",
                hovermode="x unified"
            )

            st.plotly_chart(
                fig_sales,
                use_container_width=True
            )

            # 11. CUSTOMER FORECAST CHART
    

            st.subheader("5. Daily Customer Forecast")

            fig_customers = px.line(
                result_df,
                x="Date",
                y="Predicted_Customers",
                markers=True,
                title="Predicted Daily Customers",
                template="plotly_white"
            )

            fig_customers.update_layout(
                xaxis_title="Date",
                yaxis_title="Predicted Customers",
                hovermode="x unified"
            )

            st.plotly_chart(
                fig_customers,
                use_container_width=True
            )

            st.divider()

        
            # 12. DETAILED RESULTS
        

            st.subheader("6. Detailed Prediction Results")

            display_columns = [
                "Date",
                "Store",
                "Predicted_Sales",
                "Predicted_Customers"
            ]

            display_df = result_df[
                display_columns
            ].copy()

            st.dataframe(
                display_df,
                use_container_width=True
            )

    
            # 13. DOWNLOAD RESULTS


            st.subheader("7. Download Predictions")

            csv_data = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Prediction Results",
                data=csv_data,
                file_name=(
                    f"rossmann_store_{selected_store}_predictions.csv"
                ),
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"An error occurred while generating predictions: {e}"
        )

else:

    st.info(
        "Select a store and upload a CSV file "
        "to generate predictions."
    )


# 14. FOOTER


st.divider()

st.caption(
    "Rossmann Store Sales Forecasting | "
    "Machine Learning Project"
)
