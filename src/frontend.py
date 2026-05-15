import requests
import pandas as pd
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Food Waste Prediction Dashboard",
    layout="wide"
)

st.title("AI-Powered Food Waste Prediction and Smart Inventory Optimization")
st.write(
    "This dashboard connects to the FastAPI backend to predict retail demand, "
    "estimate waste risk, monitor drift, and support retraining."
)

menu = st.sidebar.selectbox(
    "Select Page",
    [
        "Predict Demand",
        "Latest Sales",
        "Prediction Logs",
        "Model Info",
        "Drift Reports",
        "Pipeline Actions"
    ]
)


def api_get(endpoint):
    try:
        response = requests.get(f"{API_URL}{endpoint}", timeout=20)
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"API connection error: {e}")
        return None


def api_post(endpoint, payload=None):
    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=60
        )
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"API connection error: {e}")
        return None


if menu == "Predict Demand":

    st.header("Predict Product Demand and Waste Risk")

    col1, col2 = st.columns(2)

    with col1:
        product_name = st.text_input(
            "Product / Department",
            value="Dept_1"
        )

        category = st.selectbox(
            "Store Type / Category",
            ["A", "B", "C", "General"]
        )

        store_id = st.number_input(
            "Store ID",
            min_value=1,
            max_value=100,
            value=1
        )

        day_of_week = st.selectbox(
            "Day of Week",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][x]
        )

        is_weekend = 1 if day_of_week in [5, 6] else 0

        is_holiday = st.selectbox(
            "Is Holiday?",
            [0, 1]
        )

    with col2:
        promotion = st.selectbox(
            "Promotion Active?",
            [0, 1]
        )

        temperature = st.number_input(
            "Temperature",
            value=20.0
        )

        current_stock = st.number_input(
            "Current Stock",
            min_value=0,
            value=100
        )

        unit_price = st.number_input(
            "Unit Price",
            min_value=0.0,
            value=10.0
        )

        expiry_days = st.number_input(
            "Expiry Days",
            min_value=1,
            value=5
        )

        waste_quantity = st.number_input(
            "Current Waste Quantity",
            min_value=0,
            value=20
        )

    if st.button("Predict Demand"):

        payload = {
            "product_name": product_name,
            "category": category,
            "store_id": int(store_id),
            "day_of_week": int(day_of_week),
            "is_weekend": int(is_weekend),
            "is_holiday": int(is_holiday),
            "promotion": int(promotion),
            "temperature": float(temperature),
            "current_stock": int(current_stock),
            "unit_price": float(unit_price),
            "expiry_days": int(expiry_days),
            "waste_quantity": int(waste_quantity)
        }

        response = api_post("/predict-demand", payload)

        if response is not None and response.status_code == 200:
            result = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Prediction completed successfully.")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Predicted Demand",
                    result.get("predicted_demand")
                )

                c2.metric(
                    "Current Stock",
                    result.get("current_stock")
                )

                c3.metric(
                    "Waste Risk",
                    result.get("waste_risk")
                )

                st.subheader("Inventory Recommendation")
                st.write(result.get("recommendation"))

                st.write(
                    "Recommended Order Quantity:",
                    result.get("recommended_order_quantity")
                )

                st.json(result)
        else:
            st.error("Prediction request failed. Make sure FastAPI is running.")


elif menu == "Latest Sales":

    st.header("Latest Sales Records")

    limit = st.slider(
        "Number of records",
        min_value=5,
        max_value=100,
        value=20
    )

    response = api_get(f"/latest-sales?limit={limit}")

    if response is not None and response.status_code == 200:
        data = response.json()

        if len(data) == 0:
            st.info("No sales records found.")
        else:
            st.dataframe(pd.DataFrame(data), use_container_width=True)


elif menu == "Prediction Logs":

    st.header("Prediction Logs")

    limit = st.slider(
        "Number of logs",
        min_value=5,
        max_value=100,
        value=20
    )

    response = api_get(f"/prediction-logs?limit={limit}")

    if response is not None and response.status_code == 200:
        data = response.json()

        if len(data) == 0:
            st.info("No prediction logs found yet.")
        else:
            st.dataframe(pd.DataFrame(data), use_container_width=True)


elif menu == "Model Info":

    st.header("Latest Model Information")

    response = api_get("/model-info")

    if response is not None and response.status_code == 200:
        st.json(response.json())


elif menu == "Drift Reports":

    st.header("Drift Detection Reports")

    limit = st.slider(
        "Number of drift reports",
        min_value=5,
        max_value=100,
        value=20
    )

    response = api_get(f"/drift-reports?limit={limit}")

    if response is not None and response.status_code == 200:
        data = response.json()

        if len(data) == 0:
            st.info("No drift reports found yet.")
        else:
            st.dataframe(pd.DataFrame(data), use_container_width=True)


elif menu == "Pipeline Actions":

    st.header("Pipeline Actions")

    st.write(
        "Use these buttons to run MLOps pipeline actions through FastAPI."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Pipeline"):
            response = api_post("/run-pipeline")

            if response is not None:
                st.json(response.json())

    with col2:
        if st.button("Run Drift Check"):
            response = api_post("/drift-check")

            if response is not None:
                st.json(response.json())

    with col3:
        if st.button("Auto Retrain"):
            response = api_post("/auto-retrain")

            if response is not None:
                st.json(response.json())