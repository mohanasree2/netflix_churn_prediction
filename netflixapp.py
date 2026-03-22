import streamlit as st
import pandas as pd
import pickle

# ── Load model, scaler, encoders ──────────────────────────────────────────────
with open("netflix_churn_best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("netflix_churn_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("netflix_churn_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Netflix Churn Prediction", page_icon="🎬")

st.title("🎬 Netflix Churn Prediction")
st.markdown("Fill in the user details below and click **Predict** to check churn risk.")
st.divider()

# ── User Profile ──────────────────────────────────────────────────────────────
st.subheader("👤 User Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age            = st.number_input("Age", min_value=18, max_value=64, value=35)
    gender         = st.selectbox("Gender", ["Male", "Female", "Other"])
    country        = st.selectbox("Country", ["India", "USA", "Canada", "Brazil",
                                              "France", "Australia", "UK",
                                              "Japan", "Germany", "Spain"])

with col2:
    subscription   = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    monthly_fee    = st.number_input("Monthly Fee ($)", min_value=7.99, max_value=15.99, value=12.99)
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "UPI"])

with col3:
    primary_device = st.selectbox("Primary Device", ["Mobile", "Laptop", "Tablet", "Smart TV"])
    devices_used   = st.selectbox("Devices Used", [1, 2, 3])
    favorite_genre = st.selectbox("Favorite Genre", ["Action", "Comedy", "Drama",
                                                      "Horror", "Romance",
                                                      "Sci-Fi", "Thriller", "Documentary"])

st.divider()

# ── Viewing Behavior ──────────────────────────────────────────────────────────
st.subheader("📺 Viewing Behavior")

col4, col5, col6 = st.columns(3)

with col4:
    account_age    = st.number_input("Account Age (months)", min_value=1,   max_value=59,  value=24)
    avg_watch      = st.number_input("Avg Watch Time (min)", min_value=10,  max_value=299, value=90)
    sessions_week  = st.number_input("Sessions per Week",    min_value=1,   max_value=19,  value=5)

with col5:
    binge_sessions = st.number_input("Binge Sessions",       min_value=0,   max_value=14,  value=3)
    completion     = st.number_input("Completion Rate (%)",  min_value=30,  max_value=99,  value=70)
    rating         = st.number_input("Avg Rating Given",     min_value=1.0, max_value=5.0, value=3.5, step=0.1)

with col6:
    content_inter  = st.number_input("Content Interactions",          min_value=0, max_value=49, value=15)
    rec_click      = st.number_input("Recommendation Click Rate (%)", min_value=0, max_value=99, value=40)
    days_login     = st.number_input("Days Since Last Login",         min_value=0, max_value=59, value=10)

st.divider()

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🎯 Predict Churn"):

    # Build input dataframe
    input_data = pd.DataFrame([{
        "age"                       : age,
        "gender"                    : gender,
        "country"                   : country,
        "account_age_months"        : account_age,
        "subscription_type"         : subscription,
        "monthly_fee"               : monthly_fee,
        "payment_method"            : payment_method,
        "primary_device"            : primary_device,
        "devices_used"              : devices_used,
        "favorite_genre"            : favorite_genre,
        "avg_watch_time_minutes"    : avg_watch,
        "watch_sessions_per_week"   : sessions_week,
        "binge_watch_sessions"      : binge_sessions,
        "completion_rate"           : completion,
        "rating_given"              : rating,
        "content_interactions"      : content_inter,
        "recommendation_click_rate" : rec_click,
        "days_since_last_login"     : days_login,
    }])

    # Encode categorical columns
    cat_cols = ["gender", "country", "subscription_type",
                "payment_method", "primary_device", "favorite_genre"]

    for col in cat_cols:
        le  = encoders[col]
        val = input_data[col].iloc[0]
        input_data[col] = le.transform([val]) if val in le.classes_ else [0]

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    label        = encoders["churned"].inverse_transform([prediction])[0]

    # ── Result ────────────────────────────────────────────────────────────────
    st.divider()
    if label == "Yes":
        st.error("⚠️ This user is likely to CHURN.")
    else:
        st.success("✅ This user is NOT likely to churn.")
    try:
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(input_scaled)[0]
            churn_prob = proba[1] * 100
            st.metric(label="Churn Probability", value=f"{churn_prob:.1f}%")
    except Exception:
        pass