import os
import pickle
import boto3
import streamlit as st

from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
# Initialize embeddings
embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)
def main():
    st.header("Chat with PDF 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Store embeddings for the PDF file
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
            st.write("Embeddings loaded from disk.")
        else:
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
            st.write("Embeddings computed and stored.")

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            # Perform similarity search
            docs = vector_store.similarity_search(query=query, k=3)

            # Define the Bedrock LLM
            llm = Bedrock(
                model_id="meta.llama3-8b-instruct-v1:0",
                client=bedrock
            )

            # Define the prompt template
            prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
                    You are an AI assistant specializing in summarizing and answering questions from uploaded documents.
                    Use the provided context from the document to generate a concise and accurate answer.
                    Keep your answers brief and to the point. Be kind and respectful.


                    - If the answer involves a list (e.g., top 5 colleges in a city), provide it in a numbered or bullet-point format.
                    - If the answer is directly available in the context, quote it explicitly.
                    - If the context is insufficient, state "The context provided does not contain sufficient information to answer this question."
                    - Keep your response under 150 words unless otherwise requested.
                    - Maintain clarity and precision.

                    Context: {context}
                    Question: {question}
                    Answer:"""
            )

            # Create the LLMChain
            llm_chain = LLMChain(llm=llm, prompt=prompt_template)

            # Create a StuffDocumentsChain with document_variable_name
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="context"
            )

            # Create the RetrievalQA chain
            qa_chain = RetrievalQA(
                retriever=vector_store.as_retriever(),
                combine_documents_chain=combine_documents_chain,
                input_key="query",  # Ensure input_key matches your input variable
                output_key="answer" # Optional: specify output key
            )

            # Generate a response
            response = qa_chain.run({"query": query})  # Pass input as a dictionary
            st.write(response)

if __name__ == '__main__':
    main()
