import os
import logging
import requests
import json
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document, ChatMessage,SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

class QAModule:
    """A module for answering questions based on PDF content using OpenAI and FAISS for retrieval-augmented generation (RAG)."""

    def __init__(self, model_name: str = "gpt-4o-mini", embedding_model: str = "text-embedding-ada-002"):
        """
        Initializes the QAModule with a language model and embedding model for question answering.

        Args:
            model_name (str): The name of the OpenAI model to use for question answering.
            embedding_model (str): The name of the embedding model for vector store indexing.
        """
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OpenAI API key is not set. Please set it as an environment variable.")
        
        # Initialize LLM for question answering
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        
        # Initialize embedding model and vector store
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        self.vector_store = None  # Initialized later with build_index()

    def build_index(self, text: str):
        """
        Splits and indexes the provided text into vectorized chunks for retrieval.
        
        Args:
            text (str): The text content to index.
        """
        try:
            #documents = [Document(page_content=chunk) for chunk in self.text_splitter.split_text(text)]
            documents = [Document(page_content=text) ]

            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logging.info("Successfully built vector index for provided text.")
        except Exception as e:
            logging.exception("Error building vector index: %s", e)
            raise RuntimeError("Failed to build vector index.")

    def retrieve_relevant_chunks(self, question: str, k: int = 3) -> List[Document]:
        """
        Retrieves the top-k relevant chunks for a given question.

        Args:
            question (str): The question to find relevant chunks for.
            k (int): The number of top relevant chunks to retrieve.

        Returns:
            List[Document]: A list of Documents containing the relevant chunks.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Call 'build_index' first.")

        try:
            return self.vector_store.similarity_search(question, k=k)
        except Exception as e:
            logging.exception("Error during chunk retrieval: %s", e)
            return []

    def call_deepseek(prompt):
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "model": "deepseek-r1:latest",  # Ensure this matches your model
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception for bad status codes

            result = response.json()
            return result['response']

        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return None
    def answer_question(self,text_1_doc: str, text_2_transcripts: str,text_3_ticket: str, question: str) -> str:
        """
        Generates an answer based on retrieved chunks.

        Args:
            question (str): The question to answer.

        Returns:
            Optional[str]: The generated answer or None if an error occurs.
        """
        # relevant_chunks = self.retrieve_relevant_chunks(question)
        # if not relevant_chunks:
        #     logging.warning("No relevant chunks found for the question: %s", question)
        #     return "Relevant chunk is not available for given query"

        # Combine chunks into a single context for the LLM prompt
        #context = "\n".join([chunk.page_content for chunk in relevant_chunks])

        # Formulate the prompt with RAG structure
        # prompt_template = PromptTemplate(
        #     template="Context: {context}\n\nQuestion: {question}\nAnswer:",
        #     input_variables=["context", "question"]
        # )


        prompt_template = PromptTemplate(
            template="""Context: You are a software support engineer helping users solve technical problems. I'll provide you with:

1. HISTORICAL TICKETS: {historical_tickets}
2. SOFTWARE DOCUMENTATION: {software_docs}
3. TRAINING MATERIALS: {training_transcripts}
4. CURRENT USER TICKET: {user_ticket}

Review all information, but focus primarily on the historical tickets as they contain actual problem descriptions and step-by-step solutions (usually marked with "Steps" or "Investigation").

For the current user ticket, please:
- Analyze the issue carefully
- Provide a clear, step-by-step solution
- Reference relevant historical tickets, documentation, or training materials that informed your solution
- Format your response similar to the historical tickets (with clear problem description and labeled resolution steps)

Be thorough but concise in your explanation, focusing on practical steps the user can follow.

        Answer:""",
            input_variables=["historical_tickets", "software_docs", "training_transcripts", "user_ticket"]
        )
        prompt = prompt_template.format(historical_tickets=text_3_ticket, software_docs=text_1_doc, training_transcripts=text_2_transcripts, user_ticket=question)
        system_message = SystemMessage(content="If the information is not directly available in the context, respond with 'Data Not Available'.")

        # Create the user message with the prompt
        user_message = ChatMessage(role="user", content=prompt)

        # Compile messages for the ChatOpenAI model
        messages = [system_message, user_message]
        
        try:
            response = self.llm(messages)
            #response=self.call_deepseek(prompt)
            answer_text = response.content.strip()
            logging.info("Generated response for question: %s", question)
            return answer_text
        except Exception as e:
            logging.exception("Error generating answer at llm call: %s", e)
            return "Data Not Available in answer question block"


    def answer_questions(self, text_1_doc: str, text_2_transcripts: str,text_3_ticket: str, question: str) -> str:
        """
        Answers a list of questions based on the indexed text.

        Args:
            text (str): The text content to index and search.
            questions (List[str]): A list of questions to answer.

        Returns:
            Dict[str, Optional[str]]: A dictionary mapping questions to their answers.
        """
        try:
            #self.build_index(text)  # Build vector store for the provided text
            results = self.answer_question(question)
            logging.info("Successfully answered all questions.")
            return results
        except Exception as e:
            logging.exception("Error in answering questions while building the index: %s", e)
            return "Data Not Available: answer_questions "
