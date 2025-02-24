import os
import tempfile
import gc
from dataclasses import dataclass
from typing import List, Generator
from git import Repo
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.chains import RetrievalQA
from langchain.schema import Document
import yt_dlp
import torch
import psutil
import logging
import shelve
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.schema import AIMessage
from viz import DataVisualizer
from teach import main2

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom cache implementation using shelve
class CustomCache:
    def __init__(self, cache_file):
        self.cache_file = cache_file
    
    def get(self, key):
        with shelve.open(self.cache_file) as db:
            return db.get(key, None)
    
    def set(self, key, value):
        with shelve.open(self.cache_file) as db:
            db[key] = value

# Initialize custom cache
custom_cache = CustomCache('llm_cache.db')

@dataclass
class ChatbotConfig:
    chunk_size: int = 150  # Smaller chunk size
    chunk_overlap: int = 10  # Smaller overlap
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"  # Lightweight model
    llm_model: str = "phi3.5"  # Use tiny LLM model
    batch_size: int = 8  # Smaller batch size
    max_documents: int = 300  # Reduced maximum number of documents
    supported_extensions: tuple = (".py", ".js", ".md", ".txt", ".pdf", ".docx", ".css", ".html", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".rb", ".sh", ".yaml", ".json", ".xml", ".sql", ".csv", ".tsv", ".xls", ".xlsx", ".pptx", ".ppt", ".odt", ".ods", ".odp", ".odg", ".odf", ".rtf", ".tex", ".log", ".ini", ".cfg", ".conf", ".env", ".yml", ".toml", ".properties")

class MemoryManager:
    @staticmethod
    def check_memory_usage():
        """Monitor memory usage and cleanup if necessary."""
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        if memory > 2000:  # Lower threshold to 2GB
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False

    @staticmethod
    def batch_generator(items: list, batch_size: int) -> Generator:
        """Generate batches from a list of items."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

class DocumentLoader:
    @staticmethod
    def load_file(file_path: str, config: ChatbotConfig) -> List[Document]:
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            logger.debug(f"Loading file: {file_path}")
            
            # Use appropriate loader based on file type
            if file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
            
            docs = loader.load()
            logger.debug(f"Document content: {[doc.page_content for doc in docs]}")
            MemoryManager.check_memory_usage()
            
            logger.debug(f"Loaded {len(docs)} documents from file: {file_path}")
            return docs[:config.max_documents]  # Limit number of documents
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return []

class DocumentProcessor:
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        # Initialize embeddings with lower cache size
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.model_name,
            cache_folder=tempfile.gettempdir(),  # Use temp directory for cache
            encode_kwargs={'batch_size': config.batch_size}
        )

    def process_documents(self, docs: List[Document]) -> FAISS:
        """Process documents in batches to manage memory usage."""
        all_splits = []
        
        logger.debug("Splitting documents into chunks...")
        # Split documents into chunks
        for doc_batch in MemoryManager.batch_generator(docs, self.config.batch_size):
            splits = self.text_splitter.split_documents(doc_batch)
            logger.debug(f"Document chunks: {[split.page_content for split in splits]}")
            all_splits.extend(splits)
            MemoryManager.check_memory_usage()

        logger.debug(f"Total chunks created: {len(all_splits)}")
        
        # Create FAISS index in batches
        vector_store = None
        logger.debug("Creating FAISS index...")
        for batch in MemoryManager.batch_generator(all_splits, self.config.batch_size):
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                batch_store = FAISS.from_documents(batch, self.embeddings)
                vector_store.merge_from(batch_store)
            MemoryManager.check_memory_usage()
        
        return vector_store

class DocumentChatbot:
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.processor = DocumentProcessor(config)
        self.vector_store = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")

    def load_directory(self, directory: str) -> List[Document]:
        """Load documents from directory in batches."""
        docs = []
        for root, _, files in os.walk(directory):
            for file_name in files:
                if len(docs) >= self.config.max_documents:
                    break
                
                file_path = os.path.join(root, file_name)
                if any(file_path.endswith(ext) for ext in self.config.supported_extensions):
                    batch_docs = DocumentLoader.load_file(file_path, self.config)
                    docs.extend(batch_docs)
                    MemoryManager.check_memory_usage()
        
        return docs[:self.config.max_documents]

    def initialize_qa_chain(self, docs: List[Document]):
        set_llm_cache(InMemoryCache())  # Use in-memory cache for LLM
        if not docs:
            raise ValueError("No documents provided")
        
        # Process documents and create vector store
        self.vector_store = self.processor.process_documents(docs)
        
        # Initialize LLM with memory-efficient settings
        llm = ChatOllama(
            model=self.config.llm_model,
            temperature=0.1,  # Lower temperature for efficiency
            cache=True,
            num_ctx=1024  # Further reduced context window
        )
        
        # Create QA chain with memory-efficient settings
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 8}  # Reduced number of retrieved documents
            ),
            return_source_documents=True,
            verbose=False  # Reduced logging
        )
        
        MemoryManager.check_memory_usage()

    def query(self, question: str) -> str:
        if not self.qa_chain:
            raise RuntimeError("Chatbot not initialized with documents")
    
        try:
            past_conversations = self.memory.load_memory_variables({})["chat_history"]
            formatted_query = f"Previous conversation:\n{past_conversations}\nUser: {question}"

            response = self.qa_chain.invoke({"query": formatted_query}) 

            # Ensure `response` is a dictionary
            if isinstance(response, dict):
                result = response.get("result", "No answer found.")
                sources = response.get("source_documents", [])[:1]  # Limit sources displayed
            
                formatted_response = result.strip()
            if sources:
                formatted_response += "\n\nSources:"
                for i, doc in enumerate(sources, 1):
                    source_name = getattr(doc.metadata, 'source', f'Document {i}')
                    formatted_response += f"\n{i}. {source_name}"

        
                self.memory.save_context({"query": question}, {"output": formatted_response})

                return formatted_response
        

            self.memory.save_context({"query": question}, {"output": str(response)})
            return str(response)

        except Exception as e:
            return f"Error processing query: {str(e)}"

        finally:
            MemoryManager.check_memory_usage()

    
    

    def query_directly_to_llm(self, question: str) -> str:
        set_llm_cache(InMemoryCache())  # Enable caching

    # Load conversation history
        past_conversations = self.memory.load_memory_variables({})["chat_history"]

    # Limit history to the last 5 exchanges to avoid context overflow
        if isinstance(past_conversations, list):
            past_conversations = "\n".join(past_conversations[-5:])  # Convert list to string

    # Format prompt for better understanding
        formatted_query = f"""
        Conversation History:
        {past_conversations if past_conversations else "No prior conversation"}
    
        User: {question}
        AI:"""

    # Debugging: Print what is being sent to the model
        print("Formatted Query Sent to LLM:", formatted_query)

    # Initialize ChatOllama
        llm = ChatOllama(
            model=self.config.llm_model,
            temperature=0.3,  # Slightly increased for natural responses
            cache=True,
            num_ctx=1024  # Reduced context window
        )

    # Invoke the LLM and ensure safe handling of response
        response = llm.invoke(formatted_query)

    # Extract response content safely
        if isinstance(response, AIMessage):
            response_text = response.content.strip()
        else:
            response_text = str(response).strip()  # Fallback for unexpected types

    # Debugging: Print the raw response
        print("Raw LLM Response:", response_text)

    # Save conversation to memory
        self.memory.save_context({"query": question}, {"output": response_text})

        return response_text  # Return cleaned response


    

def main():
    # Initialize with memory-efficient settings
    config = ChatbotConfig()
    chatbot = DocumentChatbot(config)
    
    print("\n=== Memory-Efficient Document Chatbot ===")
    print("1. Upload file")
    print("2. Browse local directory")
    print("3. Load Git repository")
    print("4. Load YouTube video")
    print("5. Directly query LLM")
    print("6 for data vizualization")
    print("7 for vizualization for teachers")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        docs = []
        if choice == "1":
            file_path = input("Enter file path: ").strip()
            docs = DocumentLoader.load_file(file_path, config)
        elif choice == "2":
            directory = input("Enter directory path: ").strip()
            docs = chatbot.load_directory(directory)
        elif choice == "3":
            repo_url = input("Enter Git repository URL: ").strip()
            with tempfile.TemporaryDirectory() as temp_dir:
                repo = Repo.clone_from(repo_url, temp_dir)
                if repo:
                    docs = chatbot.load_directory(repo.working_tree_dir)
        elif choice == "4":
            youtube_url = input("Enter YouTube video URL: ").strip()
            transcription = main1(youtube_url)
            docs = [Document(page_content=transcription, metadata={"source": "YouTube Video"})]
        elif choice == "5":
            while True:
                question = input("\nQuestion: ").strip()
                if question.lower() == 'exit':
                    break
                if question:
                    answer = chatbot.query_directly_to_llm(question)
                    print("\nAnswer:", answer)
                    MemoryManager.check_memory_usage()
        elif choice == "6":
            file_path = input("Enter the file path (CSV/XLSX): ")
            visualizer = DataVisualizer(file_path)
            visualizer.run_visualization()
            return

        elif choice == "7":
            main2()
            return

            
        else:
            print("Invalid choice")
            return

        if not docs:
            print("No documents loaded")
            return

        chatbot.initialize_qa_chain(docs)
        print("\nChatbot ready! Type 'exit' to quit")
        
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() == 'exit':
                break
            if question:
                answer = chatbot.query(question)
                print("\nAnswer:", answer)
                MemoryManager.check_memory_usage()

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()