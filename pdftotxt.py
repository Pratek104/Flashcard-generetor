from PyPDF2 import PdfReader

# Load the PDF file
reader = PdfReader('NPL.pdf')

# Initialize an empty string to store the text
full_text = ""

# Extract text from all pages
for page in reader.pages:
    text = page.extract_text()
    if text:  # Ensure that text is not None
        full_text += text

# Split the text into a list of words and limit to the first 10,000 words
words = full_text.split()
limited_text = ' '.join(words[:10000])

# Print or return the extracted 10,000 words
