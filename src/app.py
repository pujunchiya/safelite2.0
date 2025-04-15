import logging
import os
import streamlit as st
import yaml
from dotenv import load_dotenv

from qa_agent import QAModule


load_dotenv()

# Ensure the 'log' directory exists
log_dir = os.path.join(os.getcwd(), "..", "log")  # Adjust path as needed
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "logger.log")),
        logging.StreamHandler(),
    ],
)


def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        config_path = os.path.join(os.getcwd(), "..", config_path) #adjust path relative to current working directory.
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        return None


def main(question: str):
    """Parses PDF, answers questions, and returns the response."""
    # file_path_1_docs=r"C:\SAFELITE\SAFELITE\processed_data_by_folder\1.Safelite documents_processed.txt"
    # file_path_2_transcripts=r"C:\SAFELITE\SAFELITE\processed_data_by_folder\2.Meeting Transcription_processed.txt"
    # file_path_3_ticket=r"C:\SAFELITE\SAFELITE\processed_data_by_folder\cd3.Ticket Resolution_processed.txt"

    import os

    # Get the directory of the current script (the src folder)
    src_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the data folder (one level up from src, then into 'data')
    data_dir = os.path.join(os.path.dirname(src_dir), "processed_data_by_folder")

    # Define the file paths using os.path.join for platform independence
    file_path_1_docs = os.path.join(data_dir, "1.Safelite documents_processed.txt")
    file_path_2_transcripts = os.path.join(data_dir, "2.Meeting Transcription_processed.txt")
    file_path_3_ticket = os.path.join(data_dir, "cd3.Ticket Resolution_processed.txt")

    # Now you can use these file paths in your code
    print(f"Documents file path: {file_path_1_docs}")
    print(f"Transcripts file path: {file_path_2_transcripts}")
    print(f"Ticket file path: {file_path_3_ticket}")

    # Example of opening and reading the documents file:
    try:
        with open(file_path_1_docs, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            print(f"First line of documents file: {first_line.strip()}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path_1_docs}")
    try:
        with open(file_path_1_docs, 'r', encoding='utf-8') as file:
            content_1_doc = file.read()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path_1_docs}")
        content_1_doc= ""  # Return an empty string

    try:
        with open(file_path_2_transcripts, 'r', encoding='utf-8') as file:
            content_2_transcripts = file.read()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path_2_transcripts}")
        content_2_transcripts= ""  # Return an empty string

    try:
        with open(file_path_3_ticket, 'r', encoding='cp1252', errors='ignore') as file:
            content_3_ticket = file.read()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path_3_ticket}")
        content_3_ticket= ""  # Return an empty string

    try:
  # Answer the question based on the PDF content
        logging.info("Answering the question")
        qa_agent = QAModule()
        answer = qa_agent.answer_question(content_1_doc,content_2_transcripts,content_3_ticket,question)

        return answer

    except Exception as e:
        logging.exception("An unexpected error occurred after calling QA module: %s", e)
        return f"An error occurred: {e}"


st.title("Safelite Help")

lv_question = st.text_area("Enter your question about Ticket", height=100)

if st.button("Submit"):

        try:
            logging.info("Starting App")
            response = main(lv_question)
            st.write("Response:", response)
            logging.info("App finished execution")



        except Exception as e:
            logging.exception("An unexpected error occurred in main call: %s", e)
            st.write(f"An error occurred: {e}")
