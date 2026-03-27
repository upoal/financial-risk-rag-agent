import streamlit as st
import requests

# Config 
API_URL = "https://financial-risk-rag-agent-v1.onrender.com"

st.set_page_config(
    page_title="Financial Risk Q&A Agent",
    page_icon="🏦",
    layout="centered"
)

# Header
st.title("Financial Risk Q&A Agent")
st.caption("Powered by RAG · Sources: Basel III, IFRS 9, Banxico, Santander")
st.divider()

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "This agent answers questions about financial risk regulations "
        "using retrieved excerpts from official documents. "
        "Every answer includes source citations and a confidence score."
    )
    st.divider()
    st.header("Loaded Documents")
    st.markdown("""
    - 📄 Basel III Full Framework
    - 📄 Basel III Reforms Summary
    - 📄 IFRS 9 Financial Instruments
    - 📄 Banxico Circular 3/2012
    - 📄 IFA Annual Financial Report 2025
    """)
    st.divider()
    k = st.slider("Sources to retrieve (k)", min_value=1, max_value=10, value=5)
    st.caption("Higher k = more context, slower response")

# Example Questions
st.subheader("Example Questions")
examples = [
    "What are the capital requirements under Basel III?",
    "How does IFRS 9 define expected credit loss?",
    "What are the liquidity coverage ratio requirements?",
    "How does Banxico regulate credit risk provisions?",
]

cols = st.columns(2)
for i, example in enumerate(examples):
    if cols[i % 2].button(example, use_container_width=True):
        st.session_state.selected_question = example

st.divider()

# Input 
question = st.text_area(
    "Ask a question about financial risk regulations:",
    value=st.session_state.get("selected_question", ""),
    placeholder="e.g. What is the minimum CET1 ratio under Basel III?",
    height=100
)

submit = st.button("Ask", type="primary", use_container_width=True)

# Response
if submit and question.strip():
    with st.spinner("Retrieving from documents and generating answer..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "k": k},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.Timeout:
            st.error("Request timed out. The API may be waking up — wait 30 seconds and try again.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Check that the backend is running.")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    #  Confidence Indicator 
    confidence = data["confidence"]
    low_confidence = data["low_confidence"]

    if low_confidence:
        st.error("🔴 Not Confident — insufficient information in loaded documents.")
    elif confidence >= 0.55:
        st.success("🟢 Confident")
    elif confidence >= 0.45:
        st.info("🟡 Semi-Confident")
    else:
        st.error("🔴 Not Confident")

    # Answer 
    st.subheader("Answer")
    st.write(data["answer"])
    st.caption(f"Response time: {data['response_time_ms']}ms")

    # Sources
    st.divider()
    st.subheader("📎 Sources (Audit Trail)")

    for source in data["sources"]:
        doc_name = source["document"].split("\\")[-1].split("/")[-1]
        score = source["score"]

        with st.expander(f"[{source['source_id']}] {doc_name} — Page {source['page']}"):
            st.write(f"Relevance: {source['score']:.2f}")

elif submit and not question.strip():
    st.warning("Please enter a question.")