import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
import os
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
import final_score
import ast
import test3 
import pandas as pd
import tempfile
from llama_index.core.node_parser import SimpleNodeParser
from io import BytesIO
import matplotlib.pyplot as plt
import Training
import pickle
import numpy as np
openai.api_key = st.secrets["OPENAI_API_KEY"]
api_base = "https://pro.aiskt.com/v1"
openai.base_url = api_base
st.set_page_config(page_title="PE-GPT", page_icon="💎", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the Power electronic robot🤖")
st.info( "Hello, I am a robot designed specifically for DAB!", icon="🤟")
with st.sidebar:
  st.markdown("<h1 style='color: #FF5733;'>PE-GPT (v2.0)</h1>", unsafe_allow_html=True)
  st.markdown('---')
  #st.markdown('\n- SPS:\n- EPS\n- DPS\n- TPS\n- 5DOF')
  st.markdown('\n- PE-GPT (v2.0) supports the design of modulation strategies for the dual active bridge converter.')
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
    with st.spinner(text="Loading and indexing  docs – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader("database").load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", temperature=0.1,system_prompt="You are now an expert in the power electronics industry, and you are proficient in various modulation methods(SPS,EPS,DPS,TPS,5DOF) of dual active bridge.Please answer the questions based on the documents I have provided you and your own understanding  .Make sure your answers are professional and accurate -- don't hallucinate."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
def load_data1():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader("database1").load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", system_prompt="You are now an expert in the power electronics industry, and you are proficient in various modulation methods of dual active bridge.Please provide modulation to user according to my prompt .keep your answer follow the rules I told u-- don't hallucinate."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
def load_data2():
    with st.spinner(text="Loading and indexing the  docs – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader("introduction").load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", temperature=0.1))
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

file_type = st.sidebar.selectbox("Select file type", ("vp", "vs", "iL"))

# 创建文件上传器
uploaded_file = st.sidebar.file_uploader("Upload file", key="file_uploader")

# 使用一个按钮来确认文件上传，并指定文件类型
if st.sidebar.button("Confirm Upload"):
    if uploaded_file is not None:
        # 根据选择的文件类型处理文件
        if file_type == "vp":
            vp = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')
        elif file_type == "vs":
            vs = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')
        elif file_type == "iL":
            iL = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')

        # 提示用户文件已上传并处理
        st.sidebar.write(f"{file_type} file uploaded successfully.")

# 检查所有文件是否都已上传并处理
if 'vp' in locals() and 'vs' in locals() and 'iL' in locals():
    inputs = np.concatenate((vp.T[1:, :, None], vs.T[1:, :, None]), axis=-1)
    states = iL.T[1:, :, None]
    st.write("Inputs shape:", inputs.shape)
    st.write("States shape:", states.shape)


index0 = load_data0()  
chat_engine = index0.as_chat_engine(chat_mode="context")
index1 = load_data1()  
chat_engine1 = index1.as_chat_engine()
index2 = load_data2()  
chat_engine2 = index2.as_chat_engine(chat_mode="context")
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
    if any(keyword in prompt.lower() for keyword in ["consideration","recommend","modulation","design","scheme","designs","desire","solution","goal","strategy"]):
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
            Current_Stress,nZVS,nZCS,P,pos,plot,M=test3.PINN(st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.M)
            Current_Stress1,nZVS1,nZCS1,P1,pos1,plot1,M1=test3.PINN(st.session_state.Uin,st.session_state.Uo,300,st.session_state.M)
            Current_Stress_sps,nZVS_sps,nZCS_sps,P_sps, pos_sps, plot_sps, M_sps=test3.PINN(st.session_state.Uin,st.session_state.Uo,st.session_state.P,"SPS")
            Current_Stress_sps1,nZVS_sps1,nZCS_sps1,P_sps1, pos_sps1, plot_sps1, M_sps1=test3.PINN(st.session_state.Uin,st.session_state.Uo,300,"SPS")
            Answer0=test3.answer(pos,st.session_state.M,Current_Stress,nZVS,nZCS,P,M)
            Answer1=test3.answer1(pos1,st.session_state.M,Current_Stress1,nZVS1,nZCS1,P1,M1)
            Answer2=test3.answer(pos_sps,"SPS",Current_Stress_sps,nZVS_sps,nZCS_sps,P_sps,M_sps)
            Answer3=test3.answer1(pos_sps1,"SPS",Current_Stress_sps1,nZVS_sps1,nZCS_sps1,P_sps1,M_sps1,st.session_state.M)
            st.write(Answer0)
            st.image(plot)
            st.write(Answer1)
            st.image(plot1)
            st.write(Answer2)
            st.image(plot_sps)
            st.write(Answer3)
            st.image(plot_sps1)
            message = {"role": "assistant", "content": Answer0,"images": [plot]}
            message1 = {"role": "assistant", "content": Answer1,"images": [plot1]}
            message2 = {"role": "assistant", "content": Answer2,"images": [plot_sps]}
            message3 = {"role": "assistant", "content": Answer3,"images": [plot_sps1]}
            st.session_state.messages.append(message)
            st.session_state.messages.append(message1)
            st.session_state.messages.append(message2)
            st.session_state.messages.append(message3)
    elif any(keyword in prompt for keyword in ["high", "big", "large","But","seems","satisfy","wider","requirements"]):
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
            Current_Stress,nZVS,nZCS,P,pos,plot,M=test3.PINN(st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.M)
            Current_Stress1,nZVS1,nZCS1,P1,pos1,plot1,M1=test3.PINN(st.session_state.Uin,st.session_state.Uo,300,st.session_state.M)
            Answer0=test3.answer(pos,st.session_state.M,Current_Stress,nZVS,nZCS,P,M)
            Answer1=test3.answer1(pos1,st.session_state.M,Current_Stress1,nZVS1,nZCS1,P1,M1)
            st.write(Answer0)
            st.image(plot)
            st.write(Answer1)
            st.image(plot1)
            message = {"role": "assistant", "content": Answer0,"images": [plot]}
            message1 = {"role": "assistant", "content": Answer1,"images": [plot1]}
            st.session_state.messages.append(message)
            st.session_state.messages.append(message1)
    elif "Here you go" in prompt:
         with st.chat_message("assistant"):
             with st.spinner("Thinking..."):
                plot,test_loss,val_loss=Training.Training_PINN(inputs,states)
                reply= "Retraining is done. The mean absolute error is improved from {:.3f} to {:.3f}. The predicted waveform and experimental waveform are shown below ".format(test_loss,val_loss)
                st.write(reply)
                st.image(plot)
                message = {"role": "assistant", "content": reply,"images": [plot]}
                st.session_state.messages.append(message)
    elif "experimental data" in prompt:
      with st.chat_message("assistant"):
             with st.spinner("Thinking..."):
                reply = 'Yes, I can. Please upload your data through pickle file and I will retrain the physics-informed model embedded in PE-GPT.The pickle file should be an array arr with its first element arr[0] as "inputs", and the second element arr[1] as "states".  The shape of "inputs" is bs x seqlen x ndim_inp, denoting batch size, sequence length, and dimension of inputs, respectively. The shape of "outputs" is bs x seqlen x ndim_out, denoting batch size, sequence length, and dimension of outputs, respectively.'
                st.write(reply)
                message = {"role": "assistant", "content": reply}
                st.session_state.messages.append(message)
    else:
         with st.chat_message("assistant"):
             with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt, messages_history)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)
