import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatGroq model with the API key from environment variables
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv('GROQ_API_KEY'),  # Fetch the API key from .env
    model_name="llama-3.1-70b-versatile"
)

# Load page content from the specified URL
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Nepal")
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
    keep the question at the firt line and answer at the another line,
    give 3 spaceline for the another flashcard



    ### VALID JSON (NO PREAMBLE):    
    """
)

# Create the LLM chain using the limited text
chain_extract = prompt_extract | llm 
res = chain_extract.invoke(input={'limited_text': limited_text})

# Print the result
print(res)
