import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import json
import io


st.write("Please make sure you always take a clear picture, with the name visible!")


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

    verified_names = []
    review_names = []

    total = len(uploaded_files)


    for index, uploaded_file in enumerate(uploaded_files, start=1):

        st.write(f"File {index} of {total}")

        image_bytes = uploaded_file.getvalue()

        image = Image.open(io.BytesIO(image_bytes))


        prompt = """
        Identify the student who submitted this assignment.

        Return ONLY valid JSON.
        Do not use markdown.
        Do not include explanations.

        The JSON must have exactly these fields:

        {
            "name": "student name",
            "confidence": 0.0
        }

        Confidence must be a number between 0 and 1.

        Ignore teacher names, comments, and other names.
        Do not grade the assignment.
        Do not provide feedback.
        """


        try:

            gemini_response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[
                    image,
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )


            data = json.loads(gemini_response.text)

            name = data["name"]
            confidence = data["confidence"]


            if confidence >= 0.85:

                verified_names.append(name)

            else:

                review_names.append({
                    "name": name,
                    "confidence": confidence,
                    "image": image_bytes
                })


        except json.JSONDecodeError:

            review_names.append({
                "name": "Invalid AI response",
                "confidence": 0,
                "image": image_bytes
            })


        except Exception as e:

            review_names.append({
                "name": f"Error: {e}",
                "confidence": 0,
                "image": image_bytes
            })


    verified_names.sort(key=str.lower)


    st.divider()

    st.subheader("✅ Verified Students")

    if verified_names:
        for name in verified_names:
            st.success(name)
    else:
        st.write("No automatically verified names.")



    st.divider()

    st.subheader("⚠️ Needs Review")


    if review_names:

        for index, student in enumerate(review_names):

            st.image(student["image"])

            st.write(
                f"AI suggestion: **{student['name']}**"
            )

            st.write(
                f"Confidence: {student['confidence']:.0%}"
            )


            if st.button(
                f"Accept {student['name']}",
                key=f"accept_{index}"
            ):
                verified_names.append(student["name"])
                st.success(
                    f"Added {student['name']}"
                )

    else:
        st.write("Nothing needs review.")
