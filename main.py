import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
import os
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
import final_score
import ast
import PINN
openai.api_key = st.secrets["OPENAI_API_KEY"]
api_base = "https://pro.aiskt.com/v1"
openai.base_url = api_base
st.set_page_config(page_title="Chat with the Power electronic robot", page_icon="💎", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the Power electronic robot🤖, powered by LlamaIndex 🙂")
st.info( "Hello, I am a robot designed specifically for converters!", icon="🤟")

clear_button=st.sidebar.button('Clear Conversation',key='clear')
with open('./prompt.txt', 'r') as file:
    content1 = file.read()
if clear_button or "messages" not in st.session_state:  # Initialize the chat messages history
    st.session_state.messages = [{"role": "user", "content": content1},
                                 {"role": "assistant", "content": "OK,I understand."}
                                 ]
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the buck-boost docs – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader("data").load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", temperature=0.1,system_prompt="You are now an expert in the power electronics industry, and you are proficient in various modulation methods of dual active bridge. Keep your answers technical and fact-based -- don't hallucinate."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index


index = load_data()
chat_engine = index.as_chat_engine( chat_mode="context")
modulation="X"
for message in st.session_state.messages[2:]:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    messages_history = [
    ChatMessage(role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT, content=m["content"])
    for m in st.session_state.messages
]
    if "consideration" in prompt.lower():
      with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt, messages_history)
            answer_list = ast.literal_eval(response.response)
            best_modulation=final_score.recommend_modulation(answer_list)
            modulation=best_modulation
            response1="According to your requirements, I recommend you to use the {} modulation strategy".format(best_modulation)
            st.write(response1)
            st.session_state.messages.append({"role": "assistant", "content": response1})
            messages_history = [
                ChatMessage(role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT,
                            content=m["content"])
                for m in st.session_state.messages
            ]
            response = chat_engine.chat("Please introduce this control strategy",messages_history)
            st.write(response.response)      
    elif "Uin" in prompt:
      with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st.write(modulation)
            response = chat_engine.chat(prompt, messages_history)
            answer_list1 = ast.literal_eval(response.response)
            Uin, Uo, Prated, fsw = answer_list1   
            D0,current_stress,efficiency=PINN.pinn(Uin,Uo,Prated,fsw,modulation)
            reply="The optimal D0 is designed to be {} and the current stress performance is shown with the following figure. At rated power level, the current stress is {}A.The efficiency performance is shown with the following figure . At rated power level, the  efficiency is {}.".format(D0,current_stress,efficiency)
            st.write(reply)
            col1,col2=st.columns(2)
            with col1:
              st.image("current stress.png")
            with col2:
              st.image("efficiency.png")
            message = {"role": "assistant", "content": reply}
            st.session_state.messages.append(message)
    else:
         with st.chat_message("assistant"):
             with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt,messages_history)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
