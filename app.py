
import streamlit as st
import pickle
import numpy as np


# Load saved model
model = pickle.load(open("resume_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))


st.title("🤖 AI Resume Screening & Job Matching System")

st.write("Enter your resume details to find the suitable job role")


resume_text = st.text_area(
    "Paste Your Resume Here"
)


if st.button("Analyze Resume"):

    if resume_text.strip() != "":

        # Convert resume into vector
        resume_vector = tfidf.transform([resume_text])

        # Prediction
        prediction = model.predict(resume_vector)

        # Score
        score = np.max(model.predict_proba(resume_vector)) * 100


        st.success("Analysis Completed!")

        st.write("### Recommended Job Role:")
        st.write(prediction[0])

        st.write("### Matching Score:")
        st.write(str(round(score,2)) + "%")

    else:
        st.warning("Please enter resume details")
