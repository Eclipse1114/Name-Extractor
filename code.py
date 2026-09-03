import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import json
import io


# ============================================================
# PAGE
# ============================================================

st.write(
    "Please make sure you always take a clear picture, "
    "with the student's name visible!"
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload assignment images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

submit = st.button("Extract Names")


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


try:
    client = get_gemini_client()

except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    st.stop()


# ============================================================
# PROCESS ASSIGNMENTS
# ============================================================

if submit and uploaded_files:

    st.write("Getting Student Names...")

    verified_names = []
    review_names = []

    total = len(uploaded_files)


    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------

    for index, uploaded_file in enumerate(uploaded_files, start=1):

        st.write(f"File {index} of {total}")

        image_bytes = uploaded_file.getvalue()

        try:
            image = Image.open(
                io.BytesIO(image_bytes)
            )

        except Exception as e:

            review_names.append({
                "name": f"Could not open image: {e}",
                "confidence": 0,
                "image": image_bytes
            })

            continue


        # ----------------------------------------------------
        # AI PROMPT
        # ----------------------------------------------------

        prompt = """
Identify the student's name on this assignment.

The student's name must actually be visible in the image.
Do not guess or infer a name.

Ignore:
- Teacher names
- Teacher comments
- Names appearing in the assignment questions
- Names in examples
- Names belonging to other people

Do not grade the assignment.
Do not provide feedback.

Return ONLY valid JSON.
Do not use markdown.
Do not include explanations.

The JSON must contain exactly these fields:

{
    "name": "student name",
    "confidence": 0.0
}

Rules for confidence:
- confidence must be a number between 0 and 1
- use a high confidence only when the student's name is clearly visible
- use a low confidence when the name is unclear, partially visible, or uncertain
- if no student name can be found, return an empty string for name and a confidence of 0
"""


        # ----------------------------------------------------
        # SEND TO GEMINI
        # ----------------------------------------------------

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


            # ------------------------------------------------
            # PARSE JSON
            # ------------------------------------------------

            data = json.loads(
                gemini_response.text
            )


            # ------------------------------------------------
            # VALIDATE RESPONSE
            # ------------------------------------------------

            name = data.get("name")
            confidence = data.get("confidence")


            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "AI did not return a valid student name."
                )


            if not isinstance(confidence, (int, float)):
                raise ValueError(
                    "AI did not return a numeric confidence."
                )


            confidence = float(confidence)

            # Keep confidence safely between 0 and 1
            confidence = max(
                0.0,
                min(1.0, confidence)
            )


            # ------------------------------------------------
            # SORT INTO VERIFIED / REVIEW
            # ------------------------------------------------

            if confidence >= 0.85:

                verified_names.append(
                    name.strip()
                )

            else:

                review_names.append({
                    "name": name.strip(),
                    "confidence": confidence,
                    "image": image_bytes
                })


        # ----------------------------------------------------
        # INVALID JSON
        # ----------------------------------------------------

        except json.JSONDecodeError:

            review_names.append({
                "name": "Invalid AI response",
                "confidence": 0,
                "image": image_bytes
            })


        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        except Exception as e:

            review_names.append({
                "name": f"Error: {e}",
                "confidence": 0,
                "image": image_bytes
            })


    # ========================================================
    # SORT VERIFIED NAMES
    # ========================================================

    verified_names.sort(
        key=str.lower
    )


    # ========================================================
    # VERIFIED STUDENTS
    # ========================================================

    st.divider()

    st.subheader("✅ Verified Students")

    if verified_names:

        for name in verified_names:
            st.success(name)

    else:

        st.write(
            "No automatically verified names."
        )


    # ========================================================
    # NEEDS REVIEW
    # ========================================================

    st.divider()

    st.subheader("⚠️ Needs Review")

    if review_names:

        for student in review_names:

            st.image(
                student["image"]
            )

            st.write(
                f"AI suggestion: **{student['name']}**"
            )

            st.write(
                f"Confidence: {student['confidence']:.0%}"
            )

    else:

        st.write(
            "No assignments need manual review."
        )
