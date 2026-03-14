import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=1000
)

# Page Config
st.set_page_config(
    page_title="AI Research Paper Summarizer",
    page_icon="🧠",
    layout="wide"
)

# Header
st.title("🧠 AI Research Paper Summarizer")
st.markdown("Summarize AI research papers using **Google Gemini + LangChain**")

st.divider()

# Sidebar Controls
st.sidebar.header("⚙️ Settings")

paper_input = st.sidebar.selectbox(
    "Select Research Paper",
    [
        "Attention is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners"
    ]
)

style_input = st.sidebar.selectbox(
    "Summary Style",
    ["Bullet Points", "Paragraph", "Key Takeaways"]
)

length_input = st.sidebar.selectbox(
    "Summary Length",
    ["Short", "Medium", "Long"]
)

uploaded_pdf = st.sidebar.file_uploader("Upload Research Paper PDF (optional)", type="pdf")

st.sidebar.markdown("---")
st.sidebar.info("Tip: Upload a PDF for more accurate summaries.")

# Prompt Template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI research assistant."),
    ("human",
     """
     Summarize the research paper "{paper}".

     Style: {style}
     Length: {length}

     Also provide:
     - Key contributions
     - Important concepts
     - Real-world applications
     """)
])

# Layout
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📄 Paper Information")
    st.info(paper_input)

with col2:
    st.subheader("📊 Summary Settings")
    st.write("Style:", style_input)
    st.write("Length:", length_input)

st.divider()

# Generate Summary
if st.button("🚀 Generate AI Summary", use_container_width=True):

    progress = st.progress(0)

    with st.spinner("Analyzing research paper..."):

        prompt = chat_template.invoke({
            "paper": paper_input,
            "style": style_input,
            "length": length_input
        })

        progress.progress(50)

        response = llm.invoke(prompt)

        progress.progress(100)

    st.success("Summary Generated Successfully!")

    st.subheader("📑 AI Summary")
    st.markdown(response.content)

    # Download button
    st.download_button(
        "📥 Download Summary",
        response.content,
        file_name="research_summary.txt"
    )

st.divider()

st.caption("Built with ❤️ using Streamlit, LangChain, and Gemini AI")