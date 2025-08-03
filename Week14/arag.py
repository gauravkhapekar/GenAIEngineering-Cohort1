import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from tavily import TavilyClient
import wikipedia
import arxiv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import tempfile
import shutil

# ----------------------------
# Streamlit App Configuration
# ----------------------------
st.set_page_config(page_title="Agentic RAG with Tavily + OpenRouter", layout="wide")
st.title("🔍 Agentic RAG with Tavily + GPT-4 (OpenRouter) + HuggingFace Embeddings")

# ----------------------------
# API Key Inputs
# ----------------------------
st.sidebar.header("API Keys")
openrouter_key = st.sidebar.text_input("OpenRouter API Key", type="password")
tavily_key = st.sidebar.text_input("Tavily API Key", type="password")

# ----------------------------
# Query Examples
# ----------------------------
st.sidebar.header("📝 Query Examples")
st.sidebar.markdown("""
**The system will automatically choose the best source:**

• "Who is Einstein?" → Wikipedia  
• "Latest research on quantum computing" → ArXiv  
• "Today's weather" → Web Search  
• Questions about uploaded PDFs → RAG first, then fallback

**Available Tools:**
- 📄 PDF RAG (your documents)
- 📚 Wikipedia (encyclopedic info)
- 🔬 ArXiv (academic papers)
- 🌐 Web Search (current info)
""")

# HuggingFace model for embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

retriever = None

if openrouter_key and tavily_key:
    try:
        # Set API Keys
        os.environ["OPENAI_API_KEY"] = openrouter_key
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        os.environ["TAVILY_API_KEY"] = tavily_key

        # ----------------------------
        # Initialize Tavily + LLM
        # ----------------------------
        tavily_client = TavilyClient(api_key=tavily_key)

        def tavily_search(query: str) -> str:
            """Search the web using Tavily and return formatted results"""
            try:
                response = tavily_client.search(query)
                results = response.get("results", [])
                
                if not results:
                    return "No search results found."
                
                formatted_results = []
                for i, result in enumerate(results[:5], 1):
                    title = result.get("title", "No title")
                    content = result.get("content", "No content")
                    url = result.get("url", "No URL")
                    formatted_results.append(f"{i}. {title}\n{content}\nSource: {url}\n")
                
                return "\n".join(formatted_results)
            except Exception as e:
                return f"Error searching: {str(e)}"

        def wikipedia_search(query: str) -> str:
            """Search Wikipedia for encyclopedic information"""
            try:
                # Set language to English and limit results
                wikipedia.set_lang("en")
                
                # Search for articles
                search_results = wikipedia.search(query, results=3)
                
                if not search_results:
                    return "No Wikipedia articles found for this query."
                
                # Get the first article
                try:
                    page = wikipedia.page(search_results[0])
                    summary = wikipedia.summary(search_results[0], sentences=4)
                    
                    result = f"Wikipedia Article: {page.title}\n"
                    result += f"Summary: {summary}\n"
                    result += f"URL: {page.url}\n"
                    
                    # Add alternative articles if available
                    if len(search_results) > 1:
                        result += f"\nRelated articles: {', '.join(search_results[1:])}"
                    
                    return result
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    options = e.options[:5]  # Get first 5 options
                    return f"Multiple articles found. Did you mean: {', '.join(options)}"
                    
                except wikipedia.exceptions.PageError:
                    return f"Wikipedia page not found for '{query}'"
                    
            except Exception as e:
                return f"Error searching Wikipedia: {str(e)}"

        def arxiv_search(query: str) -> str:
            """Search ArXiv for academic papers"""
            try:
                # Create search client
                client = arxiv.Client()
                
                # Search with limit
                search = arxiv.Search(
                    query=query,
                    max_results=5,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                
                results = []
                for i, result in enumerate(client.results(search), 1):
                    paper_info = f"{i}. Title: {result.title}\n"
                    paper_info += f"Authors: {', '.join([author.name for author in result.authors])}\n"
                    paper_info += f"Published: {result.published.strftime('%Y-%m-%d')}\n"
                    paper_info += f"Abstract: {result.summary[:300]}...\n"
                    paper_info += f"ArXiv URL: {result.entry_id}\n"
                    paper_info += f"PDF URL: {result.pdf_url}\n"
                    results.append(paper_info)
                
                if not results:
                    return "No academic papers found on ArXiv for this query."
                
                return "\n".join(results)
                
            except Exception as e:
                return f"Error searching ArXiv: {str(e)}"

        def smart_router(query: str) -> str:
            """Intelligently route queries to the most appropriate tool"""
            query_lower = query.lower()
            
            # Keywords that strongly suggest ArXiv search
            arxiv_keywords = [
                'research', 'paper', 'study', 'academic', 'scientific', 'publication',
                'latest research', 'recent research', 'research on', 'studies on',
                'academic paper', 'scientific paper', 'peer review', 'journal',
                'findings', 'breakthrough', 'advancement', 'discovery'
            ]
            
            # Keywords that suggest Wikipedia
            wikipedia_keywords = [
                'who is', 'what is', 'define', 'definition', 'biography', 'history of',
                'founded', 'born', 'died', 'invented', 'discovered', 'theory of',
                'concept of', 'principle of'
            ]
            
            # Check for ArXiv keywords
            if any(keyword in query_lower for keyword in arxiv_keywords):
                try:
                    result = arxiv_search(query)
                    return f"SOURCE:ARXIV||{result}"
                except Exception as e:
                    fallback = tavily_search(query)
                    return f"SOURCE:WEB||ArXiv search failed: {str(e)}. Fallback result:\n{fallback}"
            
            # Check for Wikipedia keywords  
            elif any(keyword in query_lower for keyword in wikipedia_keywords):
                try:
                    result = wikipedia_search(query)
                    return f"SOURCE:WIKIPEDIA||{result}"
                except Exception as e:
                    fallback = tavily_search(query)
                    return f"SOURCE:WEB||Wikipedia search failed: {str(e)}. Fallback result:\n{fallback}"
            
            # Default to web search for everything else
            else:
                result = tavily_search(query)
                return f"SOURCE:WEB||{result}"

        # Create a single smart routing tool
        smart_tool = Tool(
            name="smart_search",
            func=smart_router,
            description="Intelligently searches across Wikipedia, ArXiv, and web sources based on query type."
        )

        # Initialize LLM (via OpenRouter)
        llm = ChatOpenAI(
            model="openai/gpt-4o", 
            temperature=0,
            openai_api_base="https://openrouter.ai/api/v1"
        )

        # Create agent with the smart routing tool
        tools = [smart_tool]
        
        # Use standard ReAct prompt which handles parsing correctly
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

        # ----------------------------
        # PDF Upload + RAG Setup
        # ----------------------------
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files for RAG context (optional)", 
            type=["pdf"], 
            accept_multiple_files=True
        )

        if uploaded_files:
            try:
                all_docs = []
                temp_dir = tempfile.mkdtemp()
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load and process each PDF
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)

                if all_docs:
                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, 
                        chunk_overlap=200
                    )
                    texts = text_splitter.split_documents(all_docs)

                    # HuggingFace embeddings
                    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                    db = FAISS.from_documents(texts, embeddings)
                    retriever = db.as_retriever(search_kwargs={"k": 5})

                    st.success(f"{len(uploaded_files)} PDF(s) uploaded and indexed for retrieval!")
                
                # Clean up temp files
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")

        # ----------------------------
        # Query Input
        # ----------------------------
        query = st.text_input("Enter your query")

        if st.button("Run Query"):
            if query:
                try:
                    answer_source = "web"  # Default to web search
                    
                    if retriever:
                        # First, try to get an answer from RAG
                        rag_prompt = ChatPromptTemplate.from_template("""
                        Answer the question based only on the following context. If the context doesn't contain 
                        enough information to answer the question, respond with "INSUFFICIENT_CONTEXT".

                        Context: {context}

                        Question: {input}
                        
                        Answer:""")
                        
                        document_chain = create_stuff_documents_chain(llm, rag_prompt)
                        retrieval_chain = create_retrieval_chain(retriever, document_chain)
                        
                        rag_result = retrieval_chain.invoke({"input": query})
                        rag_answer = rag_result["answer"]
                        
                        # Check if RAG found relevant information
                        if "INSUFFICIENT_CONTEXT" not in rag_answer and len(rag_answer.strip()) > 20:
                            # RAG found relevant info
                            answer_source = "pdf"
                            st.subheader("📄 RAG Response (from uploaded PDFs)")
                            st.write(rag_answer)
                            
                            # Show actual source documents used
                            if "context" in rag_result:
                                st.markdown("**📋 Sources from PDFs:**")
                                for i, doc in enumerate(rag_result["context"], start=1):
                                    source_info = f"Page: {doc.metadata.get('page', 'N/A')}"
                                    file_name = doc.metadata.get('source', 'PDF')
                                    if file_name:
                                        file_name = os.path.basename(file_name)
                                    st.write(f"{i}. File: {file_name}, {source_info}")
                                    st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    
                    if answer_source == "web":
                        # Direct routing without misleading messages
                        query_lower = query.lower()
                        
                        # Keywords that strongly suggest ArXiv search
                        arxiv_keywords = [
                            'research', 'paper', 'study', 'academic', 'scientific', 'publication',
                            'latest research', 'recent research', 'research on', 'studies on',
                            'academic paper', 'scientific paper', 'peer review', 'journal',
                            'findings', 'breakthrough', 'advancement', 'discovery'
                        ]
                        
                        # Keywords that suggest Wikipedia
                        wikipedia_keywords = [
                            'who is', 'what is', 'define', 'definition', 'biography', 'history of',
                            'founded', 'born', 'died', 'invented', 'discovered', 'theory of',
                            'concept of', 'principle of'
                        ]
                        
                        # Direct routing logic with proper messaging
                        if any(keyword in query_lower for keyword in arxiv_keywords):
                            # Use ArXiv
                            if retriever:
                                st.info("📄 No relevant information found in uploaded PDFs. Searching ArXiv for academic papers...")
                            st.subheader("🔬 ArXiv Research Papers")
                            try:
                                arxiv_result = arxiv_search(query)
                                st.write(arxiv_result)
                            except Exception as e:
                                st.error(f"ArXiv search failed: {str(e)}")
                                st.info("Falling back to web search...")
                                web_response = agent_executor.invoke({"input": query})
                                st.subheader("🌐 Web Search Response (Fallback)")
                                st.write(web_response["output"])
                        
                        elif any(keyword in query_lower for keyword in wikipedia_keywords):
                            # Use Wikipedia
                            if retriever:
                                st.info("📄 No relevant information found in uploaded PDFs. Searching Wikipedia...")
                            st.subheader("📚 Wikipedia Information")
                            try:
                                wiki_result = wikipedia_search(query)
                                st.write(wiki_result)
                            except Exception as e:
                                st.error(f"Wikipedia search failed: {str(e)}")
                                st.info("Falling back to web search...")
                                web_response = agent_executor.invoke({"input": query})
                                st.subheader("🌐 Web Search Response (Fallback)")
                                st.write(web_response["output"])
                        
                        else:
                            # Use Web Search
                            if retriever:
                                st.info("📄 No relevant information found in uploaded PDFs. Searching the web...")
                            web_response = agent_executor.invoke({"input": query})
                            response_text = web_response["output"]
                            
                            # Parse the source information if it exists
                            if "SOURCE:" in response_text and "||" in response_text:
                                source_part, content_part = response_text.split("||", 1)
                                source = source_part.replace("SOURCE:", "")
                                
                                if source == "ARXIV":
                                    st.subheader("🔬 ArXiv Research Papers")
                                    st.write(content_part)
                                elif source == "WIKIPEDIA":
                                    st.subheader("📚 Wikipedia Information")
                                    st.write(content_part)
                                else:
                                    st.subheader("🌐 Web Search Response")
                                    st.write(content_part)
                            else:
                                st.subheader("🌐 Web Search Response")
                                st.write(response_text)
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.write("Falling back to basic web search...")
                    try:
                        basic_search = tavily_search(query)
                        st.subheader("🌐 Basic Web Search Results")
                        st.write(basic_search)
                    except Exception as e2:
                        st.error(f"Search failed: {str(e2)}")
            else:
                st.warning("Please enter a query.")

    except Exception as e:
        st.error(f"Initialization error: {str(e)}")

else:
    st.warning("Please provide both OpenRouter and Tavily API keys in the sidebar to continue.")