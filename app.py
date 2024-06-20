import time 
import os
import logging
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoModelForCausalLM
from langchain_groq import ChatGroq
from footer import footer
import config


st.set_page_config(page_title="BeLegal.ai", layout="centered")

with st.sidebar:
    # st.title("BeLegal.ai")
    st.markdown("<h1 style='text-align: center; color: orange'>BeLegal.ai</h1>", unsafe_allow_html=True)
    st.image("D:\Project\Legal_LLM\legalbot.webp")
    st.text("An AI-powered Chat Bot which \nassists you with your legal works \nspecially built for Indian cases.")
    st.text("\n")
    st.text("\n")
    st.text("A product by 'BeTeam',\na team building AI products \nfor India.")
    st.text("\n")
    st.text("\n")
    st.text("Our other products")
    st.markdown("<h1 style='color: lightgreen;'>BeAware.ai</h1>", unsafe_allow_html=True)

# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)



@st.cache_resource
def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    return embeddings


embeddings = load_embeddings()
db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
print("DB created.")
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_template = """
<s>[INST]
As a legal chatbot specializing in the Indian Penal Code, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
- Respond in a bullet-point format to clearly delineate distinct aspects of the legal query.
- Each point should accurately reflect the breadth of the legal provision in question, avoiding over-specificity unless directly relevant to the user's query.
- Clarify the general applicability of the legal rules or sections mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
- Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
- Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations unless otherwise specified.
- Conclude with a brief summary that captures the essence of the legal discussion and corrects any common misinterpretations related to the topic.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
- [Detail the first key aspect of the law, ensuring it reflects general application]
- [Provide a concise explanation of how the law is typically interpreted or applied]
- [Correct a common misconception or clarify a frequently misunderstood aspect]
- [Detail any exceptions to the general rule, if applicable]
- [Include any additional relevant information that directly relates to the user's query]
</s>[INST]
"""



prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])


llm = ChatGroq(model='mixtral-8x7b-32768', temperature=0.5, max_tokens=1024, groq_api_key=config.GROQ_API_KEY)

qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=st.session_state.memory, retriever=db_retriever, combine_docs_chain_kwargs={'prompt': prompt})

def extract_answer(full_response):
    """Extracts the answer from the LLM's full response by removing the instructional text."""
    answer_start = full_response.find("Response:")
    if answer_start != -1:
        answer_start += len("Response:")
        answer_end = len(full_response)
        return full_response[answer_start:answer_end].strip()
    return full_response

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


input_prompt = st.chat_input("Say something...")
if input_prompt:
    with st.chat_message("user"):
        st.markdown(f"**You:** {input_prompt}")

    st.session_state.messages.append({"role": "user", "content": input_prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking 💡..."):
            result = qa.invoke(input=input_prompt)
            message_placeholder = st.empty()
            answer = extract_answer(result["answer"])

            # Initialize the response message
            full_response = "⚠️ **_Gentle reminder: We generally ensure precise information, but do double-check._** \n\n\n"
            for chunk in answer:
                # Simulate typing by appending chunks of the response over time
                full_response += chunk
                time.sleep(0.02)  # Adjust the sleep time to control the "typing" speed
                message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.button('🗑️ Reset All Chat', on_click=reset_conversation):
            st.experimental_rerun()