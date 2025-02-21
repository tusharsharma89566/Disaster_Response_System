import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
from datetime import datetime

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def initialize_page():
    st.set_page_config(
        page_title="Military Protocol Assistant",
        page_icon="🎖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced styling with military-appropriate dark theme
    st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --background-color: #1a1a1a;
            --secondary-bg: #2d2d2d;
            --accent-color: #4CAF50;
            --text-color: #e0e0e0;
            --border-color: #404040;
        }
        
        /* Global styles */
        .main {
            background-color: var(--background-color);
            color: var(--text-color);
            padding: 2rem;
        }
        
        .stApp {
            background-color: var(--background-color);
        }
        
        /* Header styling */
        .header-container {
            background-color: var(--secondary-bg);
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid var(--accent-color);
            margin-bottom: 2rem;
        }
        
        /* Status indicators */
        .status-online {
            color: var(--accent-color);
            font-weight: bold;
        }
        
        /* Chat interface */
        .chat-container {
            background-color: var(--secondary-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .message-box {
            background-color: #383838;
            padding: 1rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            border-left: 3px solid var(--accent-color);
        }
        
        /* Custom button styling */
        .stButton>button {
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        
        /* Input field styling */
        .stTextInput>div>div>input {
            background-color: var(--secondary-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }
        
        /* Sidebar customization */
        .css-1d391kg {
            background-color: var(--secondary-bg);
        }
        
        /* Protocol cards */
        .protocol-card {
            background-color: var(--secondary-bg);
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            margin: 0.5rem 0;
        }
        
        /* Emergency indicator */
        .emergency-status {
            background-color: #ff4444;
            color: white;
            padding: 0.5rem;
            border-radius: 4px;
            text-align: center;
            margin: 1rem 0;
        }
        
        /* Timestamp styling */
        .timestamp {
            color: #888;
            font-size: 0.8rem;
            text-align: right;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

def initialize_prompt():
    return ChatPromptTemplate.from_template("""
    You are a **Military Emergency Protocol Assistant**, responsible for providing **strictly accurate** guidance based on **official military protocol documents**.  

    **STRICT RESPONSE GUIDELINES:**  
    1. **Use ONLY the official protocol documents** to generate responses. No external assumptions, opinions, or alternative advice are allowed.  
    2. **Directly extract and present** the full, actionable steps from the provided protocols. **DO NOT** tell the user to check the procedures themselves—give them the exact details.  
    3. **Reject unnecessary, vague, or unrelated queries** by responding with:  
       ```
       "Information Not available in the database"
       ```
    4. **If the query is relevant to emergency proceduresbu t NOT found in the protocol documents, respond with:**  
       ```
       "This information is not available in the official protocols. In such cases, follow standard emergency procedures:  
       - Stay calm and assess the situation.  
       - Ensure the safety of yourself and your unit.  
       - Follow general emergency protocols as trained.  
       - Seek immediate guidance from your commanding officer or emergency response teams."
       ```
    5. **Prioritize clarity, urgency, and step-by-step execution** for emergency situations. Responses must be **fully detailed**, with no missing steps.  
    6. **Structure the response as follows (ONLY if the data is in the protocol):**  

    **Response Format:**  
    - **Immediate Actions (if applicable) → Critical steps that must be taken immediately.**  
    - **Step-by-step procedure → Fully detailed steps extracted from the protocols.**  
    - **Protocol Reference → Section, page, or source from which the information was retrieved.**  

    **Official Protocol Data:**  
    {context}  

    **Query:** {input}  
    """)




@st.cache_resource
def process_documents_background():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader("./Data")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        return FAISS.from_documents(final_documents, embeddings)
    except Exception as e:
        st.error(f"Protocol database initialization error: {str(e)}")
        return None

def display_system_status():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**System Status:** <span class='status-online'>● ONLINE</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")
    with col3:
        st.markdown("**Protocol Database:** <span class='status-online'>● ACTIVE</span>", unsafe_allow_html=True)

def main():
    initialize_page()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/8943/8943377.png", width=150)
        st.title("Soldiers Assistant")
        
        # System initialization
        with st.spinner("Initializing protocol database..."):
            vectors = process_documents_background()
            if vectors:
                st.success("✓ Systems Operational")
                st.info(f"Active Protocols: {len(vectors.docstore._dict)}")
            else:
                st.error("× Protocol Database Offline")
                return
        
        # Quick access protocols
        st.subheader("Quick Access Protocols")
        if st.button("📡 Communications Failure"):
            st.session_state.question = "What to do during communications equipment failure?"
        if st.button("🔥 Fire Hazard"):
            st.session_state.question = "How to respond to a fire outbreak in the field?"
        if st.button("💥 Explosive Threat"):
            st.session_state.question = "What are the immediate steps when encountering an explosive or bomb threat?"
        if st.button("🏥 Casualty Evacuation"):
            st.session_state.question = "How to perform a safe and efficient casualty evacuation?"
        if st.button("🚨 Ambush Response"):
            st.session_state.question = "What are the tactical steps to take during an ambush?"
        if st.button("❄️ Hypothermia & Cold Injuries"):
            st.session_state.question = "How to prevent and treat hypothermia in extreme cold conditions?"
        if st.button("🌡️ Heat Exhaustion & Dehydration"):
            st.session_state.question = "What are the signs and first aid measures for heat exhaustion?"
        if st.button("🚧 Minefield Encounter"):
            st.session_state.question = "What are the safety protocols when encountering a suspected minefield?"
        if st.button("🎯 Sniper Threat"):
            st.session_state.question = "How to react and take cover in a sniper threat situation?"
        if st.button("🦠 Biological or Chemical Attack"):
            st.session_state.question = "How to respond in case of a suspected biological or chemical attack?"
        if st.button("📍 Lost in Unfamiliar Terrain"):
            st.session_state.question = "What are the survival steps if lost in an unfamiliar environment?"
        if st.button("🔋 Power & Equipment Failure"):
            st.session_state.question = "What to do in case of critical equipment or power failure?"

    
    # Main content area
    st.markdown("""
        <div class="header-container">
            <h1>Military Protocol Assistant</h1>
            <p>Emergency Response and Field Operations Support System</p>
        </div>
    """, unsafe_allow_html=True)
    
    display_system_status()
    
    # Query input
    question = st.text_input(
        "Request Protocol Information",
        value=st.session_state.get('question', ''),
        placeholder="Enter your query or situation description...",
        key="protocol_query"
    )
    
    # Process query
    if question:
        with st.spinner("Analyzing protocols..."):
            try:
                llm = initialize_llm()
                document_chain = create_stuff_documents_chain(llm, initialize_prompt())
                retrieval_chain = create_retrieval_chain(vectors.as_retriever(), document_chain)
                
                start_time = time.process_time()
                response = retrieval_chain.invoke({'input': question})
                processing_time = time.process_time() - start_time
                
                # Display response
                st.markdown("### Protocol Response")
                st.markdown("""
                    <div class="message-box">
                        <div class="response-content">
                            {response}
                        </div>
                        <div class="timestamp">
                            Response time: {time:.2f}s
                        </div>
                    </div>
                """.format(
                    response=response['answer'].replace('\n', '<br>'),
                    time=processing_time
                ), unsafe_allow_html=True)
                
                # Reference materials
                with st.expander("📚 Reference Materials"):
                    for idx, doc in enumerate(response["context"], 1):
                        st.markdown(f"""
                            <div class="protocol-card">
                                <strong>Protocol Reference {idx}</strong><br>
                                {doc.page_content}
                            </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error("Protocol retrieval error. Initiating backup systems.")
                st.exception(e)

if __name__ == "__main__":
    main()