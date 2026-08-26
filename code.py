import streamlit as st
from google import genai
from google.genai import types
from PIL import Image


uploaded_files = st.file_uploader(
    "Upload assignment images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

submit = st.button("Extract Names")


@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


try:
    client = get_gemini_client()
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    st.stop()


if submit and uploaded_files:
    st.write("Getting Student Names...")
    x = 1
    total = len(uploaded_files)
    all_names = []

    for uploaded_file in uploaded_files:
        st.write(f"File {x} of {total}")
        image = Image.open(uploaded_file)

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[
                image,
                """
                Identify the student who submitted this assignment.

                Find the student's own name written on the assignment.
                Ignore all other names that may appear in the image.

                Return ONLY the student's name.
                If you cannot confidently identify the student's name, return UNKNOWN.

                Do not grade the assignment.
                Do not provide feedback.
                Do not explain anything.
                """
            ]
        )

        all_names.append(response.text)
        x += 1
    all_names.sort(key=str.lower)
    
    st.write("Students who submitted")
    for names in all_names:
        st.write(names)
