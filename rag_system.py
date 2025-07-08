# References:
# https://medium.com/mitb-for-all/a-guide-to-code-testing-rag-agents-without-real-llms-or-vector-dbs-51154ad920be


import os
from typing import List, Optional
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from datasets import load_dataset

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


class SimpleRAG:
    """Simple RAG implementation using LlamaIndex with Hugging Face models"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model: str = "bigscience/bloomz-560m"):
        """
        Initialize RAG system with Hugging Face models
        
        Args:
            embedding_model: Hugging Face embedding model name
            llm_model: Hugging Face language model name
        """
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.documents = []
        self.index = None
        self.query_engine = None

        self._setup_models()
    
    def _setup_models(self):
        """Setup embedding and LLM models"""
        try:
            # Setup embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model_name
            )
            

            self.llm = OpenAI(
                model_name=self.llm_model_name,
                max_tokens=256,
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            logger.info("Models setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise
    
    def load_documents_from_hf(self, dataset_name: str,
                              text_column: str = "text",
                              limit: int = 100) -> List[Document]:
        """
        Load documents from Hugging Face dataset
        
        Args:
            dataset_name: Name of the HF dataset
            text_column: Column name containing text data
            limit: Maximum number of documents to load
            
        Returns:
            List of LlamaIndex Document objects
        """
        try:

            dataset = load_dataset(dataset_name, split="train")
            
            # Convert to LlamaIndex documents
            documents = []
            for i, item in enumerate(dataset):
                if i >= limit:
                    break
                    
                if text_column in item and item[text_column]:
                    doc = Document(
                        text=str(item[text_column]),
                        metadata={"source": dataset_name, "index": i}
                    )
                    documents.append(doc)
            
            self.documents = documents
            logger.info(f"Loaded {len(documents)} documents from {dataset_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def create_index(self) -> VectorStoreIndex:
        """
        Create vector index from loaded documents
        
        Returns:
            VectorStoreIndex object
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents_from_hf first.")
        
        try:
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                embed_model=self.embed_model,
                llm=self.llm
            )
            logger.info("Vector index created successfully")
            return self.index
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def setup_retriever(self, similarity_top_k: int = 3) -> VectorIndexRetriever:
        """
        Setup document retriever
        
        Args:
            similarity_top_k: Number of similar documents to retrieve
            
        Returns:
            VectorIndexRetriever object
        """
        if not self.index:
            raise ValueError("Index not created. Call create_index first.")
        
        try:
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k
            )
            logger.info("Retriever setup completed")
            return self.retriever
            
        except Exception as e:
            logger.error(f"Error setting up retriever: {e}")
            raise
    
    def setup_query_engine(self) -> RetrieverQueryEngine:
        """
        Setup query engine for RAG
        
        Returns:
            RetrieverQueryEngine object
        """
        if not hasattr(self, 'retriever'):
            raise ValueError("Retriever not setup. Call setup_retriever first.")
        
        try:
            # Setup postprocessor to filter results
            postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
            
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                node_postprocessors=[postprocessor]
            )
            logger.info("Query engine setup completed")
            return self.query_engine
            
        except Exception as e:
            logger.error(f"Error setting up query engine: {e}")
            raise
    
    def query(self, question: str) -> str:
        """
        Query the RAG system
        
        Args:
            question: User question
            
        Returns:
            Generated response
        """
        if not self.query_engine:
            raise ValueError("Query engine not setup. Call setup_query_engine first.")
        
        try:
            response = self.query_engine.query(question)
            logger.info(f"Query processed successfully: {question[:50]}...")
            logger.info(f"{response}")
            return str(response)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def get_retrieved_documents(self, question: str) -> List[Document]:
        """
        Get documents retrieved for a question (for testing purposes)
        
        Args:
            question: User question
            
        Returns:
            List of retrieved documents
        """
        if not hasattr(self, 'retriever'):
            raise ValueError("Retriever not setup. Call setup_retriever first.")
        
        try:
            retrieved_nodes = self.retriever.retrieve(question)
            return [node.node for node in retrieved_nodes]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise