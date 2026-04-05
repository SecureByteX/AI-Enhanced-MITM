import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="MITM Detection System", layout="centered")

st.title("🔐 MITM Attack Detection System")


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)


if "safe" not in st.session_state:
    st.session_state.safe = 0
if "attack" not in st.session_state:
    st.session_state.attack = 0


st.subheader("🧪 Enter Packet Details")

col1, col2 = st.columns(2)

with col1:
    packet_size = st.number_input("Packet Size", min_value=1, value=100)
    src_port = st.number_input("Source Port", min_value=1, value=50000)

with col2:
    dst_port = st.number_input("Destination Port (Server Port)", min_value=1, value=5000)
    protocol = st.selectbox("Protocol", [6], format_func=lambda x: "TCP (6)")


if st.button("Analyze Traffic"):

    data = pd.DataFrame(
        [[packet_size, src_port, dst_port, protocol]],
        columns=['packet_size','src_port','dst_port','protocol']
    )

    
    data_scaled = scaler.transform(data)

    
    prediction = model.predict(data_scaled)[0]

    
    if prediction == 1:
        st.session_state.attack += 1
        st.error("🚨 Suspicious Traffic Detected")
    else:
        st.session_state.safe += 1
        st.success("✅ Normal Traffic")

    
    st.subheader("🔬 Explanation (Why this decision?)")

    shap_values = explainer.shap_values(data_scaled)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    features = ['packet_size','src_port','dst_port','protocol']

    explanation = pd.DataFrame({
        "Feature": features,
        "Impact": shap_vals
    }).sort_values(by="Impact", key=abs, ascending=False)

    
    st.dataframe(explanation, width='stretch')

    
    fig, ax = plt.subplots()
    ax.barh(explanation["Feature"], explanation["Impact"])
    ax.set_title("Feature Impact on Decision")
    ax.invert_yaxis()

    st.pyplot(fig)

    st.caption("Positive → Attack | Negative → Safe")

st.subheader("📊 Traffic Summary")

safe = st.session_state.safe
attack = st.session_state.attack

fig, ax = plt.subplots()
ax.bar(['Normal', 'Attack'], [safe, attack])
ax.set_title("Traffic Classification")

st.pyplot(fig)

st.write(f"✅ Normal: {safe}")
st.write(f"🚨 Suspicious: {attack}")


st.subheader("📊 Model Insight")

if st.button("Show Feature Importance"):

    features = ['packet_size','src_port','dst_port','protocol']
    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_title("Feature Importance")

    st.pyplot(fig)


if st.button("Reset"):
    st.session_state.safe = 0
    st.session_state.attack = 0
