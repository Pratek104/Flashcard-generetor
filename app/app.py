import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import json

# Load environment variables
load_dotenv()
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.1-70b-versatile",
)

# Function to extract text from the uploaded PDF file
def extract_text_from_pdf(pdf_file, word_limit=10000):
    reader = PdfReader(pdf_file)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text

    words = full_text.split()
    limited_text = ' '.join(words[:word_limit])
    
    return limited_text

from webtotxt import res

# Streamlit UI
st.title("PDF Flashcard Generator")
st.write("Upload a PDF file to extract text and generate flashcards.")

# PDF file uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        limited_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF text loaded successfully!")

        # Display the extracted text (limited to first 200 characters for readability)
        st.text_area("Extracted Text", value=limited_text, height=300)

        # Create the prompt for the LLM
        prompt_extract = PromptTemplate.from_template(
            f"""
            Give a short summary on:
            ==========================
         
            {limited_text}
            ==========================
            The scraped text is from the webpage,
            give only main info in summarized way using flashcards,
            give random flashcards anywhere from the text,
            provide only 10 flashcards completely random without repeating,
            make sure to make the question bold,
            keep the question at the first line and answer at another line,
            give 3 spacelines for the next flashcard.


            ### VALID JSON (NO PREAMBLE):  

            """
        )

        # Create the LLM chain
        chain_extract = LLMChain(llm=llm, prompt=prompt_extract)

        # Generate flashcards when the button is clicked
        if st.button("Generate Flashcards"):
            res = chain_extract.run({"limited_text": limited_text})
            st.write("", res)

            # Try to parse the response into a dictionary
            try:
                flashcards_dict = json.loads(res)  # Assume response is in JSON format
                st.subheader("Generated Flashcards")
                for key, value in flashcards_dict.items():
                    st.markdown(f"**{key}:** {value}")
            except (ValueError, SyntaxError, json.JSONDecodeError) as e:
                st.error(f"Error parsing the response: {e}")

    except Exception as e:
        st.error(f"Error loading PDF text: {e}")
else:
    st.info("Please upload a PDF file to get started.")
