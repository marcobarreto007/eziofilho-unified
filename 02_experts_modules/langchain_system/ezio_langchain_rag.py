# ezio_langchain_rag.py - Advanced RAG System for Financial Analysis
# Audit Mode: LangChain RAG with vector database
# Path: C:\Users\anapa\eziofilho-unified\02_experts_modules\langchain_system
# User: marcobarreto007
# Date: 2025-05-24 16:41:01 UTC

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# LangChain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader, 
    PDFLoader, 
    CSVLoader,
    UnstructuredURLLoader
)
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
import chromadb

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.api_config import API_KEYS

class EzioLangChainRAG:
    """Advanced RAG System for Financial Document Analysis"""
    
    def __init__(self):
        print("=" * 80)
        print("🔗 EZIOFILHO LANGCHAIN RAG SYSTEM")
        print("📚 Initializing document analysis system...")
        print("=" * 80)
        
        # Initialize components
        self.setup_embeddings()
        self.setup_vector_store()
        self.setup_llm()
        self.setup_memory()
        self.setup_tools()
        
        print("✅ RAG System Ready!")
        print("=" * 80)
        
    def setup_embeddings(self):
        """Setup embedding model"""
        print("📊 Loading embedding model...")
        
        # Use multilingual embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def setup_vector_store(self):
        """Setup vector database"""
        print("🗄️ Initializing vector store...")
        
        # Create persistent directory
        persist_directory = Path(__file__).parent / "chroma_db"
        persist_directory.mkdir(exist_ok=True)
        
        # Initialize Chroma
        self.vectorstore = Chroma(
            collection_name="financial_docs",
            embedding_function=self.embeddings,
            persist_directory=str(persist_directory)
        )
        
    def setup_llm(self):
        """Setup language model"""
        print("🤖 Loading language model...")
        
        # Check if using local or API model
        if API_KEYS.get("openai"):
            from langchain.llms import OpenAI
            self.llm = OpenAI(
                api_key=API_KEYS["openai"],
                temperature=0.7,
                model_name="gpt-3.5-turbo"
            )
        else:
            # Use local model
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            model_id = "microsoft/phi-2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
    def setup_memory(self):
        """Setup conversation memory"""
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,  # Remember last 5 exchanges
            return_messages=True
        )
        
    def setup_tools(self):
        """Setup LangChain tools"""
        self.tools = [
            Tool(
                name="Document_Search",
                func=self.search_documents,
                description="Search financial documents for information"
            ),
            Tool(
                name="Market_Analysis",
                func=self.analyze_market,
                description="Analyze market conditions and trends"
            ),
            Tool(
                name="Risk_Assessment",
                func=self.assess_risk,
                description="Assess investment risks"
            ),
            Tool(
                name="News_Summary",
                func=self.summarize_news,
                description="Summarize financial news"
            )
        ]
        
    def load_document(self, file_path: str) -> List[Dict]:
        """Load and process documents"""
        print(f"📄 Loading document: {file_path}")
        
        # Determine loader based on file type
        if file_path.endswith('.pdf'):
            loader = PDFLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
            
        # Load documents
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vectorstore.add_documents(chunks)
        
        print(f"✅ Loaded {len(chunks)} chunks from {file_path}")
        return chunks
        
    def load_urls(self, urls: List[str]) -> List[Dict]:
        """Load content from URLs"""
        print(f"🌐 Loading {len(urls)} URLs...")
        
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        
        # Split and store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        chunks = text_splitter.split_documents(documents)
        self.vectorstore.add_documents(chunks)
        
        print(f"✅ Loaded {len(chunks)} chunks from URLs")
        return chunks
        
    def search_documents(self, query: str, k: int = 5) -> str:
        """Search documents using similarity search"""
        results = self.vectorstore.similarity_search(query, k=k)
        
        if not results:
            return "No relevant documents found."
            
        response = "📚 Relevant information from documents:\n\n"
        for i, doc in enumerate(results, 1):
            response += f"{i}. {doc.page_content[:200]}...\n"
            response += f"   Source: {doc.metadata.get('source', 'Unknown')}\n\n"
            
        return response
        
    def analyze_market(self, query: str) -> str:
        """Analyze market conditions"""
        # Create market analysis chain
        template = """Based on the following context and question, provide a detailed market analysis.
        
        Context: {context}
        Question: {question}
        
        Provide analysis covering:
        1. Current market conditions
        2. Key trends
        3. Opportunities
        4. Risks
        5. Recommendations
        
        Analysis:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain.run(query)
        
    def assess_risk(self, query: str) -> str:
        """Assess investment risks"""
        template = """Analyze the investment risks based on the context and question.
        
        Context: {context}
        Question: {question}
        
        Risk Assessment should include:
        1. Market Risk
        2. Credit Risk
        3. Liquidity Risk
        4. Operational Risk
        5. Regulatory Risk
        6. Overall Risk Score (1-10)
        7. Mitigation Strategies
        
        Risk Assessment:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain.run(query)
        
    def summarize_news(self, query: str) -> str:
        """Summarize financial news"""
        template = """Summarize the financial news and insights based on the context.
        
        Context: {context}
        Question: {question}
        
        News Summary should include:
        1. Key Headlines
        2. Market Impact
        3. Affected Sectors/Assets
        4. Sentiment Analysis
        5. Trading Implications
        
        Summary:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain.run(query)
        
    def create_financial_agent(self):
        """Create ReAct agent for financial analysis"""
        
        # Agent prompt
        agent_prompt = """You are a financial analysis expert with access to various tools.
        
        You have access to the following tools:
        {tools}
        
        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        Thought: {agent_scratchpad}"""
        
        prompt = PromptTemplate(
            template=agent_prompt,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5
        )
        
        return self.agent_executor
        
    def query(self, question: str) -> str:
        """Query the RAG system"""
        
        # Use agent if available
        if hasattr(self, 'agent_executor'):
            return self.agent_executor.run(question)
        else:
            # Fallback to simple QA
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever()
            )
            return qa_chain.run(question)
            
    def interactive_session(self):
        """Run interactive RAG session"""
        print("\n💬 LangChain RAG Financial Analysis")
        print("Commands: 'load <file>', 'url <url>', 'search <query>', 'exit'")
        print("-" * 60)
        
        # Create agent
        self.create_financial_agent()
        
        while True:
            user_input = input("\n📝 You: ").strip()
            
            if user_input.lower() == 'exit':
                print("👋 Goodbye!")
                break
                
            elif user_input.startswith('load '):
                file_path = user_input[5:].strip()
                try:
                    self.load_document(file_path)
                except Exception as e:
                    print(f"❌ Error loading file: {e}")
                    
            elif user_input.startswith('url '):
                url = user_input[4:].strip()
                try:
                    self.load_urls([url])
                except Exception as e:
                    print(f"❌ Error loading URL: {e}")
                    
            elif user_input.startswith('search '):
                query = user_input[7:].strip()
                results = self.search_documents(query)
                print(f"\n🤖 Assistant: {results}")
                
            else:
                # General query
                response = self.query(user_input)
                print(f"\n🤖 Assistant: {response}")

# Advanced Document Processing

class FinancialDocumentProcessor:
    """Process financial documents for RAG"""
    
    @staticmethod
    def process_earnings_report(file_path: str) -> Dict[str, Any]:
        """Extract key metrics from earnings reports"""
        # Implementation for parsing earnings reports
        pass
        
    @staticmethod
    def process_research_report(file_path: str) -> Dict[str, Any]:
        """Extract insights from research reports"""
        # Implementation for parsing research reports
        pass
        
    @staticmethod
    def process_sec_filing(file_path: str) -> Dict[str, Any]:
        """Extract data from SEC filings"""
        # Implementation for parsing SEC filings
        pass

# Vector Database Management

class VectorDBManager:
    """Manage vector database collections"""
    
    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        
    def create_collection(self, name: str, metadata: Dict = None):
        """Create new collection"""
        return self.client.create_collection(
            name=name,
            metadata=metadata or {}
        )
        
    def list_collections(self):
        """List all collections"""
        return self.client.list_collections()
        
    def delete_collection(self, name: str):
        """Delete collection"""
        self.client.delete_collection(name=name)

# Main execution
if __name__ == "__main__":
    try:
        import torch
        
        # Create RAG system
        rag_system = EzioLangChainRAG()
        
        # Example: Load some financial documents
        # rag_system.load_document("financial_report.pdf")
        # rag_system.load_urls(["https://finance.yahoo.com/news"])
        
        # Run interactive session
        rag_system.interactive_session()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()