import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatGroq model with the API key from environment variables
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv('GROQ_API_KEY'),  # Fetch the API key from .env
    model_name="llama-3.1-70b-versatile"
)

# Streamlit UI
st.title("Webpage Flashcard Generator")
st.write("Enter a URL to scrape the content and generate flashcards.")

# User input for the URL
url = st.text_input("Enter a webpage URL", "https://en.wikipedia.org/wiki/Nepal")

# Button to trigger the scraping and flashcard generation
if st.button("Generate Flashcards"):
    try:
        # Load page content from the specified URL
        loader = WebBaseLoader(url)
        page_data = loader.load().pop().page_content

        # Limit the extracted text to 10,000 words
        words = page_data.split()  # Split the text into words
        limited_text = ' '.join(words[:10000])  # Join the first 10,000 words

        # Prepare the prompt template with the limited text
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {limited_text}
            ### INSTRUCTION:
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

        # Create the LLM chain using the limited text
        chain_extract = LLMChain(llm=llm, prompt=prompt_extract)
        res = chain_extract.run({"limited_text": limited_text})

        # Display the generated flashcards
        st.subheader("Generated Flashcards")
        st.write(res)

    except Exception as e:
        st.error(f"Error loading webpage or generating flashcards: {e}")
else:
    st.info("Please enter a URL and click the button to generate flashcards.")
