import streamlit as st # type: ignore
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import random

# --- Dummy Training Data (for demo) ---
X_train = np.array([
    [1200, 3, 2, 2005, 5000, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0],  # [sqft, bedrooms, bathrooms, year, lot, garage, city1, city2, city3, floors, pool, garden, style1, style2, style3]
    [1500, 4, 3, 2010, 6000, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    [800, 2, 1, 1995, 3000, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [2000, 5, 4, 2018, 8000, 1, 0, 0, 1, 3, 1, 1, 0, 0, 1],
    [1000, 2, 2, 2000, 4000, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0]
])
y_train = np.array([120000, 180000, 90000, 250000, 110000])
model = LinearRegression().fit(X_train, y_train)

# --- Streamlit UI ---
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #fff8f0;}
    .stButton>button {background-color: #ff7043; color: white; font-weight: bold; border-radius: 8px;}
    .stButton>button:hover {background-color: #ffab91;}
    .stSidebar {background-color: #fbe9e7;}
    .section {background: linear-gradient(90deg, #ff7043 60%, #ffd54f 100%); border-radius: 18px; padding: 1.2rem; margin-bottom: 1.5rem;}
    .section-title {color: #fff; font-size: 1.5rem; font-weight: bold;}
    .input-summary {background: #fffde7; border-radius: 10px; padding: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🏠 House Price Prediction App")
st.markdown(
    "<div class='section'><span class='section-title'>Predict your house price instantly! 🏡</span></div>",
    unsafe_allow_html=True
)
st.image("https://images.unsplash.com/photo-1568605114967-8130f3a36994?auto=format&fit=crop&w=800&q=80", use_container_width=True)

st.markdown("### 📝 Enter House Details:")

col1, col2, col3 = st.columns(3)
with col1:
    sqft = st.number_input("📏 Square Footage", min_value=300, max_value=10000, value=1200, step=50)
    year = st.number_input("📅 Year Built", min_value=1900, max_value=2025, value=2005, step=1)
    zip_code = st.text_input("🏷️ Zip Code", value="12345")
with col2:
    bedrooms = st.number_input("🛏️ Bedrooms", min_value=1, max_value=10, value=3, step=1)
    lot = st.number_input("🌳 Lot Size (sqft)", min_value=500, max_value=20000, value=5000, step=100)
    floors = st.number_input("🏢 Number of Floors", min_value=1, max_value=5, value=2, step=1)
with col3:
    bathrooms = st.number_input("🛁 Bathrooms", min_value=1, max_value=10, value=2, step=1)
    garage = st.selectbox("🚗 Garage", ["Yes", "No"])
    pool = st.selectbox("🏊 Pool", ["Yes", "No"])
    garden = st.selectbox("🌼 Garden", ["Yes", "No"])

city = st.selectbox("📍 Location", ["City Center", "Suburb", "Countryside"])
house_type = st.selectbox("🏠 House Style", ["Detached", "Semi-Detached", "Apartment"])

# One-hot encoding for city and style (for demo)
city1 = 1 if city == "City Center" else 0
city2 = 1 if city == "Suburb" else 0
city3 = 1 if city == "Countryside" else 0

style1 = 1 if house_type == "Detached" else 0
style2 = 1 if house_type == "Semi-Detached" else 0
style3 = 1 if house_type == "Apartment" else 0

garage_val = 1 if garage == "Yes" else 0
pool_val = 1 if pool == "Yes" else 0
garden_val = 1 if garden == "Yes" else 0

# --- Image Upload ---
st.markdown("### 🖼️ Upload a House Photo (optional)")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image:
    st.image(uploaded_image, caption="Your House", use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "favorites" not in st.session_state:
    st.session_state.favorites = []

if st.button("🔮 Predict Price"):
    features = np.array([[sqft, bedrooms, bathrooms, year, lot, garage_val, city1, city2, city3, floors, pool_val, garden_val, style1, style2, style3]])
    price = model.predict(features)[0]
    ci_low = price * 0.95
    ci_high = price * 1.05
    st.success(f"🏡 Estimated House Price: **${price:,.0f}**")
    st.info(f"95% Confidence Interval: ${ci_low:,.0f} - ${ci_high:,.0f}")
    if price > 200000:
        st.snow()
    st.markdown("<div class='input-summary'><b>Your Inputs:</b><br>", unsafe_allow_html=True)
    st.write({
        "Square Footage": sqft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Year Built": year,
        "Lot Size": lot,
        "Floors": floors,
        "Garage": garage,
        "Pool": pool,
        "Garden": garden,
        "Zip Code": zip_code,
        "Location": city,
        "House Style": house_type
    })
    st.markdown("</div>", unsafe_allow_html=True)
    # Save to history
    st.session_state.history.append({
        "price": price,
        "inputs": {
            "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "year": year, "lot": lot, "floors": floors, "garage": garage,
            "pool": pool, "garden": garden, "zip": zip_code,
            "city": city, "type": house_type
        }
    })

    # --- Mortgage Calculator ---
    st.markdown("### 💸 Mortgage Calculator")
    colA, colB, colC = st.columns(3)
    with colA:
        down_payment = st.number_input("Down Payment ($)", min_value=0, value=int(price*0.2), step=1000)
    with colB:
        interest = st.number_input("Interest Rate (%)", min_value=1.0, max_value=15.0, value=6.5, step=0.1)
    with colC:
        years = st.number_input("Loan Term (years)", min_value=5, max_value=40, value=30, step=1)
    loan = price - down_payment
    monthly_rate = interest/100/12
    n_payments = years*12
    if loan > 0:
        monthly_payment = loan * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
        st.success(f"Estimated Monthly Payment: **${monthly_payment:,.0f}**")
    else:
        st.info("No loan needed!")

    # --- Feature Importance ---
    st.markdown("### 📊 Feature Importance (Demo)")
    feature_names = ["sqft", "bed", "bath", "year", "lot", "garage", "city1", "city2", "city3", "floors", "pool", "garden", "style1", "style2", "style3"]
    fig, ax = plt.subplots(figsize=(8,2))
    ax.barh(feature_names, model.coef_, color="#ff7043")
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    # --- Neighborhood Info (Fake) ---
    st.markdown("### 🏘️ Neighborhood Info")
    neighborhoods = {
        "12345": {"schools": 4, "parks": 2, "crime": "Low"},
        "54321": {"schools": 2, "parks": 1, "crime": "Medium"},
        "67890": {"schools": 5, "parks": 3, "crime": "Very Low"},
    }
    info = neighborhoods.get(zip_code, {"schools": 3, "parks": 1, "crime": "Unknown"})
    st.write(f"**Schools Nearby:** {info['schools']}")
    st.write(f"**Parks Nearby:** {info['parks']}")
    st.write(f"**Crime Rate:** {info['crime']}")

    # --- Save Favorite Predictions ---
    if st.button("⭐ Save This Prediction as Favorite"):
        st.session_state.favorites.append(st.session_state.history[-1])

# --- Prediction History ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("📊 Prediction History")
    for i, h in enumerate(reversed(st.session_state.history[-5:]), 1):
        st.markdown(
            f"<div style='background:#ffe0b2; border-radius:8px; padding:0.5rem; margin-bottom:0.5rem;'>"
            f"<b>{i}. ${h['price']:,.0f}</b> | {h['inputs']}</div>",
            unsafe_allow_html=True
        )
    # --- Download Prediction as CSV ---
    df = pd.DataFrame([
        {"Price": h["price"], **h["inputs"]}
        for h in st.session_state.history
    ])
    st.download_button(
        "Download Prediction History as CSV",
        df.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )

# --- Favorite Predictions ---
if st.session_state.favorites:
    st.markdown("### ⭐ Favorite Predictions")
    for fav in st.session_state.favorites:
        st.write(f"**${fav['price']:,.0f}** | {fav['inputs']}")

# --- Sidebar Tips ---
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Home Buying Tip:**")
tips = [
    "Get pre-approved for a mortgage before you shop.",
    "Consider future resale value.",
    "Inspect the house thoroughly.",
    "Check neighborhood amenities.",
    "Budget for closing costs."
]
st.sidebar.info(random.choice(tips))
