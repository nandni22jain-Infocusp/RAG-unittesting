from rag_system import SimpleRAG

# Initialize RAG system
rag = SimpleRAG(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="EleutherAI/gpt-neo-125M"
)

# Load documents from Hugging Face dataset
documents = rag.load_documents_from_hf(
    dataset_name="squad", 
    text_column="context",  # Column containing text
    limit=50  # Number of documents to load
)

# Create vector index
index = rag.create_index()

# Setup retriever
retriever = rag.setup_retriever(similarity_top_k=3)

# Setup query engine
query_engine = rag.setup_query_engine()

# Query the system
response = rag.query("what is Notre Dame's")
print(f"response is {response}")

# Get retrieved documents
retrieved_docs = rag.get_retrieved_documents("what is Notre Dame's")
print(f"Retrieved {len(retrieved_docs)} documents")
print("First retrieved document:", retrieved_docs[0].text if retrieved_docs else "No documents found")