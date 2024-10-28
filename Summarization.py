import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from langchain import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
# Set up Google API key
genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))
api_key = os.getenv('GOOGLE_API_KEY') 
# Initialize the LLM
llm= ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=api_key)
# Load the PDF file
loader = PyMuPDFLoader("Movie_Report.pdf")
# Extract the documents
docs = loader.load()
for doc in docs:
    print(doc.page_content)
# Create a list of Document objects
document = [Document(page_content=doc.page_content) for doc in docs]
# Define the prompt template
template = '''Write a concise and short summary of the following speech in 300 words.
Speech: {document}
'''
prompt = PromptTemplate(
    input_variables=['document'],
    template=template
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
# Load the summarization chain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="document")
# Run the summarization chain on the documents
output_summary = stuff_chain.invoke(document)  # Pass the 'document' list instead of 'docs'
print(type(output_summary))  # Print the summary
print(output_summary["output_text"])