import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
import os
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
import final_score
import ast
import test2 
import pandas as pd
import tempfile
from llama_index.core.node_parser import SimpleNodeParser
from io import BytesIO
import matplotlib.pyplot as plt
openai.api_key = st.secrets["OPENAI_API_KEY"]
api_base = "https://pro.aiskt.com/v1"
openai.base_url = api_base
st.set_page_config(page_title="PE-GPT", page_icon="💎", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the Power electronic robot🤖, powered by LlamaIndex 🙂")
st.info( "Hello, I am a robot designed specifically for converters!", icon="🤟")
with st.sidebar:
  st.markdown("<h1 style='color: #FF5733;'>Optional modulation strategy</h1>", unsafe_allow_html=True)
  st.markdown('---')
  st.markdown('\n- SPS\n- EPS\n- DPS\n- TPS\n- 5DOF')
  st.markdown('---')
clear_button=st.sidebar.button('Clear Conversation',key='clear')

with open('./prompt.txt', 'r') as file:
    content1 = file.read()
if clear_button or "messages" not in st.session_state:  # Initialize the chat messages history
    st.session_state.messages = [{"role": "user", "content": content1},
                                 {"role": "assistant", "content": "OK,I understand.I will follow your instructions"}
                                 ]
@st.cache_resource(show_spinner=False)
def load_data0():
    with st.spinner(text="Loading and indexing the buck-boost docs – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader("database").load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613", temperature=0.1,system_prompt="You are now an expert in the power electronics industry, and you are proficient in various modulation methods of dual active bridge.Please answer the questions based on the documents I have provided you and your own understanding .Keep your answers technical and fact-based -- don't hallucinate."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
def load_data1():
    with st.spinner(text="Loading and indexing the buck-boost docs – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader("database1").load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613", temperature=0,system_prompt="You are now an expert in the power electronics industry, and you are proficient in various modulation methods of dual active bridge.Please provide modulation to user according to my prompt .Keep your answers technical and fact-based -- don't hallucinate."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
def load_data2():
    with st.spinner(text="Loading and indexing the buck-boost docs – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader("introduction").load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613", temperature=0.1))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
if "M" not in st.session_state:
    st.session_state.M = "X"  # 初始化M的值
if "Uin" not in st.session_state: 
   st.session_state.Uin=1
if "Uo" not in st.session_state: 
    st.session_state.Uo=1
if "P" not in st.session_state:
    st.session_state.P=1

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
  #temp_dir = tempfile.mkdtemp()
 # path = os.path.join(temp_dir, uploaded_file.name)
 # with open(path, "wb") as f:
  #  f.write(uploaded_file.getvalue())
 # documents=SimpleDirectoryReader(temp_dir).load_data()
 # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", temperature=0.1,system_prompt="You are a data analyst and you need to analyze this set of data for users"))
  #index1 = VectorStoreIndex.from_documents(documents, service_context=service_context)
  #parser = SimpleNodeParser()
  #new_nodes = parser.get_nodes_from_documents(documents)
 # for d in documents:
 #   index.insert(document=d,service_context=service_context)
index0 = load_data0()  
chat_engine = index0.as_chat_engine( chat_mode="context")
index1 = load_data1()  
chat_engine1 = index1.as_chat_engine( )
index2 = load_data2()  
chat_engine2 = index2.as_chat_engine( chat_mode="context")
for message in st.session_state.messages[2:]:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "images" in message:
          st.image(message["images"])

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    messages_history = [
    ChatMessage(role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT, content=m["content"])
    for m in st.session_state.messages
]
    if any(keyword in prompt.lower() for keyword in ["consideration","requirement","recommend","modulation","design","scheme","designs","desire","solution","goal","strategy"]):
        with st.chat_message("assistant"):
          with st.spinner("Thinking..."):
              response = chat_engine1.chat(prompt, messages_history)
              st.write(response.response)
              modulation_methods = ["SPS", "DPS", "EPS", "TPS", "5DOF"]
              first_method_found = None
              first_method_index = len(response.response)
              # 遍历每个方法，检查它是否在response.response中，并记录位置
              for method in modulation_methods:
                  index = response.response.find(method)
                  # 如果找到了方法，并且这个位置比之前记录的位置更前，就更新记录
                  if index != -1 and index < first_method_index:
                      first_method_found = method
                      first_method_index = index
                # 如果找到了一个方法，就设置st.session_state.M
              if first_method_found:
                    st.session_state.M = first_method_found
              message = {"role": "assistant", "content": response.response}
              st.session_state.messages.append(message)
    elif any(keyword in prompt for keyword in ["PE-GPT", "introduce"]):
      with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine2.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
    elif "Uin" in prompt:
      with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt, messages_history)
            answer_list1 = ast.literal_eval(response.response)
            st.session_state.Uin, st.session_state.Uo,st.session_state.P = answer_list1   
            Current_Stress,Current_Stress1,nZVS,P,pos,plot,M=test2.PINN(st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.M)
            Current_Stress_sps,Current_Stress1_sps,nZVS_sps,P_sps, pos_sps, plot_sps, M_sps=test2.PINN(st.session_state.Uin,st.session_state.Uo,st.session_state.P,"SPS")
            Answer=test2.answer(pos,st.session_state.M ,Current_Stress,Current_Stress1,nZVS,P,M)
            reply=Answer
            Answer1=test2.answer1(Current_Stress_sps,Current_Stress1_sps)
            reply1=Answer1
            st.write(reply)
            st.image(plot)
            st.write(reply1)
            st.image(plot_sps)
            message = {"role": "assistant", "content": reply,"images": [plot]}
            message1 = {"role": "assistant", "content": reply1,"images": [plot_sps]}
            st.session_state.messages.append(message)
            st.session_state.messages.append(message1)
    elif any(keyword in prompt.lower() for keyword in ["high", "big", "large","but","seems","satisfy","ZVS performance"]):
        with st.chat_message("assistant"):
          with st.spinner("Thinking..."):
              response = chat_engine.chat(prompt, messages_history)
              st.write(response.response)
              modulation_methods = ["SPS", "DPS", "EPS", "TPS", "5DOF"]
              last_method_found = None
              last_method_index = -1  # 初始化为-1，表示尚未找到

              # 反向遍历每个方法，检查它是否在response.response中，并记录位置
              for method in reversed(modulation_methods):
                    index = response.response.rfind(method)  # 使用rfind来找最后一次出现的位置
                      # 如果找到了方法，并且这个位置比之前记录的位置更后，就更新记录
                    if index != -1 and index > last_method_index:
                            last_method_found = method
                            last_method_index = index

# 如果找到了一个方法，就设置st.session_state.M
              if last_method_found:
                      st.session_state.M = last_method_found
              message = {"role": "assistant", "content": response.response}
              st.session_state.messages.append(message)
    elif "OK" in prompt:
      with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            Current_Stress,Current_Stress1,nZVS,P,pos,plot,M=test2.PINN(st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.M)
            Answer=test2.answer(pos,st.session_state.M ,Current_Stress,Current_Stress1,nZVS,P,M)
            reply=Answer
            st.write(reply)
            st.image(plot)
            message = {"role": "assistant", "content": reply,"images": [plot]}
            st.session_state.messages.append(message)
    else:
         with st.chat_message("assistant"):
             with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt,messages_history)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
