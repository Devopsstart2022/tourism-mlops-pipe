
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__),
                              "tourism_project/deployment/best_model.pkl")
    return joblib.load(model_path)

model = load_model()

# ── Encoding maps (match prep.py LabelEncoder alphabetical order) ─
TYPE_MAP         = {"Company Invited": 0, "Self Enquiry": 1}
OCCUPATION_MAP   = {"Free Lancer": 0, "Large Business Owner": 1,
                    "Salaried": 2, "Self Employed": 3, "Small Business Owner": 4}
GENDER_MAP       = {"Female": 0, "Male": 1}
PRODUCT_MAP      = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
MARITAL_MAP      = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
DESIGNATION_MAP  = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}

# ── App header ────────────────────────────────────────────────
st.title("✈️ Tourism Package Purchase Predictor")
st.markdown("**Visit with Us** | Will this customer buy the Wellness Tourism Package?")
st.divider()

# ── Input form ────────────────────────────────────────────────
st.subheader("👤 Customer Details")
col1, col2 = st.columns(2)

with col1:
    age               = st.number_input("Age", 18, 64, 35)
    type_of_contact   = st.selectbox("Type of Contact", list(TYPE_MAP.keys()))
    city_tier         = st.selectbox("City Tier", [1, 2, 3])
    occupation        = st.selectbox("Occupation", list(OCCUPATION_MAP.keys()))
    gender            = st.selectbox("Gender", list(GENDER_MAP.keys()))
    n_person          = st.number_input("Number of Persons Visiting", 1, 10, 2)
    preferred_star    = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status    = st.selectbox("Marital Status", list(MARITAL_MAP.keys()))
    designation       = st.selectbox("Designation", list(DESIGNATION_MAP.keys()))
    monthly_income    = st.number_input("Monthly Income (₹)", 5000, 100000, 25000, step=1000)

with col2:
    n_trips           = st.number_input("Number of Trips/Year", 1, 22, 3)
    passport          = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
    own_car           = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
    n_children        = st.number_input("Number of Children Visiting", 0, 5, 0)
    pitch_score       = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched   = st.selectbox("Product Pitched", list(PRODUCT_MAP.keys()))
    n_followups       = st.number_input("Number of Follow-ups", 1, 6, 3)
    duration_pitch    = st.number_input("Duration of Pitch (mins)", 5, 60, 15)

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("🔮 Predict Purchase", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "Age"                       : age,
        "TypeofContact"             : TYPE_MAP[type_of_contact],
        "CityTier"                  : city_tier,
        "DurationOfPitch"           : duration_pitch,
        "Occupation"                : OCCUPATION_MAP[occupation],
        "Gender"                    : GENDER_MAP[gender],
        "NumberOfPersonVisiting"    : n_person,
        "NumberOfFollowups"         : n_followups,
        "ProductPitched"            : PRODUCT_MAP[product_pitched],
        "PreferredPropertyStar"     : preferred_star,
        "MaritalStatus"             : MARITAL_MAP[marital_status],
        "NumberOfTrips"             : n_trips,
        "Passport"                  : passport,
        "PitchSatisfactionScore"    : pitch_score,
        "OwnCar"                    : own_car,
        "NumberOfChildrenVisiting"  : n_children,
        "Designation"               : DESIGNATION_MAP[designation],
        "MonthlyIncome"             : monthly_income,
    }])

    prediction = model.predict(input_data)[0]
    proba      = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ **LIKELY TO PURCHASE** — Probability: {proba:.1%}")
        st.balloons()
    else:
        st.warning(f"❌ **UNLIKELY TO PURCHASE** — Probability: {proba:.1%}")

    with st.expander("View Input Data"):
        st.dataframe(input_data)

st.divider()
st.caption("Tourism Package Predictor | Visit with Us | Powered by ML")
