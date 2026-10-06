import streamlit as st
import json

# Load model-package data
with open("model_data.json") as f:
    model_packages = json.load(f)

# Page config
st.set_page_config(page_title="ML Package Recommender", page_icon="🧠")

# Title
st.title("🧠 ML Package Recommender")
st.write("Select your ML / AI Model and get required Python packages with install command.")

st.divider()

# Dropdown
model = st.selectbox("Choose Model / Algorithm", list(model_packages.keys()))

# Button
if st.button("Generate Packages"):

    packages = model_packages[model]
    command = "pip install " + " ".join(packages)

    st.success("Packages Generated Successfully!")

    st.subheader("📦 Required Packages")
    for pkg in packages:
        st.write("✔️", pkg)

    st.subheader("💻 Install Command")
    st.code(command, language="bash")

    # requirements.txt content
    req_text = "\n".join(packages)

    st.download_button(
        label="📥 Download requirements.txt",
        data=req_text,
        file_name="requirements.txt",
        mime="text/plain"
    )

st.divider()

# Smart Text Input (Optional AI feature)
st.subheader("🤖 Describe Your Project (Smart Suggestion)")

user_text = st.text_input("Example: I want to build face detection system")

if user_text:
    text = user_text.lower()

    if "face" in text:
        st.info("Suggested Model: Face Detection")
    elif "image" in text:
        st.info("Suggested Model: Image Processing")
    elif "nlp" in text or "text" in text:
        st.info("Suggested Model: NLP Basic")
    elif "deep learning" in text:
        st.info("Suggested Model: CNN (Deep Learning)")
    else:
        st.warning("No suggestion found. Please use dropdown.")
