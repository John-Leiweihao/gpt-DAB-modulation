
import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
import os
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage, MessageRole
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
import Buck_plecs
import test_buck

#调用openai的API
openai.api_key = "sk-FQ9kAJr8D6aKEtZj6aC43dEbDc964a96A5D96880B7A03cA1"
api_base = "https://pro.aiskt.com/v1"
openai.base_url = api_base

#GUI界面布局
st.set_page_config(page_title="PE-GPT", page_icon="💎", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the Power electronic robot🤖")
st.info( "Hello, I am a robot specifically for power electronics design!", icon="🤟")
with st.sidebar:
  st.markdown("<h1 style='color: #FF5733;'>PE-GPT (v2.0)</h1>", unsafe_allow_html=True)
  st.markdown('---')
  #st.markdown('\n- SPS:\n- EPS\n- DPS\n- TPS\n- 5DOF')
  st.markdown('\n- PE-GPT (v2.0) supports the design of modulation strategies for the dual active bridge converter.')
  st.markdown('\n- PE-GPT (v2.0) supports the optimal design for buck converter.')
  st.markdown('---')
clear_button=st.sidebar.button('Clear Conversation',key='clear')

#GPT模型选择
client = openai.OpenAI(api_key=openai.api_key,base_url=api_base)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-0125-preview"
  
#将prompt提供给GPT模型，进行预训练
with open('prompt.txt', 'r') as file:
    content1 = file.read()
if clear_button or "messages" not in st.session_state:  # Initialize the chat messages history
    st.session_state.messages = [{"role": "user", "content": content1},
                                 {"role": "assistant", "content": "OK,I understand.I will follow your instructions and keep my answer as concisely and to the point as possible"}
                                 ]
#采用RAG方法将不同的文档分别嵌入GPT中，创建不同的GPT模型
@st.cache_resource(show_spinner=False)
def load_data0():
    with st.spinner(text="Loading and indexing  docs – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader("database").load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents(docs)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", temperature=0.1,system_prompt="You are now an expert in the power electronics industry, and you are proficient in various modulation methods(SPS,EPS,DPS,TPS,5DOF) of dual active bridge and optimzation design of buck converter.Please answer the questions based on the documents I have provided you and your own understanding  .Make sure your answers are professional and accurate -- don't hallucinate."))
        index = VectorStoreIndex(nodes, service_context=service_context)
        return index
def load_data1():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader("database1").load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents(docs)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", system_prompt="You are now an expert in the power electronics industry, and you are proficient in various modulation methods of dual active bridge.Please provide modulation to user according to my prompt .keep your answer follow the rules I told u-- don't hallucinate."))
        index = VectorStoreIndex(nodes, service_context=service_context)
        return index
def load_data2():
    with st.spinner(text="Loading and indexing the  docs – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader("introduction").load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents(docs)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", temperature=0.1))
        index = VectorStoreIndex(nodes, service_context=service_context)
        return index
def load_data3():
        docs = SimpleDirectoryReader("database_empty").load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents(docs)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-0125-preview", system_prompt="You are now an expert in the power electronics industry, and you are proficient in various modulation methods of dual active bridge.Please provide modulation to user according to my prompt .keep your answer follow the rules I told u-- don't hallucinate."))
        index = VectorStoreIndex(nodes, service_context=service_context)
        return index
index0 = load_data0()  
chat_engine = index0.as_chat_engine(similarity_top_k=7)
index1 = load_data1()  
chat_engine1 = index1.as_chat_engine(similarity_top_k=7)
index2 = load_data2()  
chat_engine2 = index2.as_chat_engine(chat_mode="context",similarity_top_k=7)
index3 = load_data3()  
chat_engine3 = index3.as_chat_engine()

#定义可能用到的变量名
initial_values = {
    "M": "X",
    "Uin": 1,
    "Uo": 1,
    "P": 1,
    "L": 1,
    "C": 1,
    "fs": 1,
    "i_ripple_lim": 1,
    "v_ripple_lim": 1,
    "i_ripple_value": 1,
    "v_ripple_value": 1,
    "i_ripple_percentage": 1,
    "v_ripple_percentage": 1,
    "iLdc": 1,
    "iL1": 1,
    "iL2": 1,
    "iL3": 1,
    "Vodc": 1,
    "Vo1": 1,
    "Vo2": 1,
    "Vo3": 1,
    "P_on": 1,
    "P_off": 1,
    "P_cond": 1,
    "vp": None,
    "vs": None,
    "iL": None
}


# 遍历字典，初始化 st.session_state 中的值
for key, value in initial_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# 创建一个选择文件类型的下拉菜单
file_type = st.sidebar.selectbox("Select file type", ("vp", "vs", "iL"))

# 创建文件上传器
uploaded_file = st.sidebar.file_uploader("Upload file", key="file_uploader")

# 使用一个按钮来确认文件上传，并指定文件类型
if st.sidebar.button("Confirm Upload"):
    if uploaded_file is not None:
        # 根据选择的文件类型处理文件
        if file_type == "vp":
            st.session_state.vp = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')
        elif file_type == "vs":
            st.session_state.vs = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')
        elif file_type == "iL":
            st.session_state.iL = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')

        # 提示用户文件已上传并处理
        st.sidebar.write(f"{file_type} file uploaded successfully.")


def determine_chat_engine(user_input,messages_history):
    # 使用 GPT 来分析输入语境
    prompt = f"""
      The input content is: "{user_input}"
      Please determine which case this belongs to:
      1. Case 0: The user is inquiring about information related to the dual active bridge converter, except for asking you to recommend a modulation method for it.
      2. Case 1:The user needs to choose or update the modulation method for the dual active bridge converter.
      3. Case 2:The user needs you to introduce yourself (PE-GPT) or  the user ask what is PE-GPT.
      You only need to understand the user's input and Return the most appropriate case..
    """
    response = chat_engine3.chat(prompt,messages_history)  # 假设 gpt_model 是你使用的 GPT 接口
    decision = response.response.strip()

    # 根据 GPT 的判断选择相应的 chat_engine
    if "0" in decision:
        return chat_engine  # 使用 index0 的 chat_engine
    elif "1" in decision:
        return chat_engine1  # 使用 index1 的 chat_engine
    elif "2" in decision:
        return chat_engine2  # 使用 index2 的 chat_engine
    else:
        return chat_engine3  # 
        
def determine_action(user_input,messages_history):
    # 使用 GPT 来分析输入语境
    prompt = f"""
      The input content is: "{user_input}"
      Please determine which action to execute:
      1. Action 0: The user provides the operating conditions of the dual active bridge(DAB) converter for the first time  and requests a design based on these conditions.
      2. Action 1:When you have recommended a new modulation method to the user and the user directly expresses the need to redesign the dual active bridge converter using the new modulation method.If the user only expresses that the current modulation method does not meet their application requirements, Action1 does not need to be executed.
      3. Action 2:The user provides the design requirements and operating conditions for the Buck converter and requests the design of the Buck converter accordingly.
      4. Action 3:The user requests you to analyze the harmonic components of the inductor current and capacitor voltage in the Buck converter.
      5. Action 4: The user requests you to validate the design results using PLECS.
      6. Action 5:The user's instruction did not execute all of the metioned actions.
      You only need to understand the user's input and Return the most appropriate action.
    """
    response = chat_engine3.chat(prompt,messages_history)  # 假设 gpt_model 是你使用的 GPT 接口
    Action = response.response.strip()
    return Action
  
#在GUI展示历史聊天对话
for message in st.session_state.messages[2:]:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "images" in message:
          st.image(message["images"])
          
# 检查所有文件是否都已上传并处理
if st.session_state.vp is not None and st.session_state.vs is not None and st.session_state.iL is not None:
    if 'is_processed' not in st.session_state:
        st.session_state.is_processed = False  # 初始化标志位

    if not st.session_state.is_processed:
        inputs = np.concatenate((st.session_state.vp.T[1:, :, None], st.session_state.vs.T[1:, :, None]), axis=-1)
        states = st.session_state.iL.T[1:, :, None]
        with st.chat_message("assistant"):
            with st.spinner("Processing data..."):
                plot, test_loss, val_loss = Training.Training_PINN(inputs, states)
                reply = "Retraining is done. The mean absolute error is improved from {:.3f} to {:.3f}. The predicted waveform and experimental waveform are shown below ".format(test_loss, val_loss)
                st.write(reply)
                st.image(plot)
                message = {"role": "assistant", "content": reply, "images": [plot]}
                st.session_state.messages.append(message)

        st.session_state.is_processed = True  # 标志位设为True，表示已处理
#用户提问框，各种提问方式以及对应回答所调用的GPT模型
if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    messages_history = [
    ChatMessage(role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT, content=m["content"])
    for m in st.session_state.messages
]
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        selected_engine = determine_chat_engine(prompt, messages_history)
        Action=determine_action(prompt,messages_history)
        st.write(Action)
        response =selected_engine.chat(prompt, messages_history)
        decision = response.response
        if decision.rfind("recommend") != -1:
            # 获取 "recommend" 之后的部分
            recommend_index = decision.rfind("recommend")
            subsequent_decision = decision[recommend_index + len("recommend"):]
            first_keyword = None
            for keyword in ["SPS", "DPS", "EPS", "TPS", "5DOF"]:
              keyword_index = subsequent_decision.find(keyword)
              if keyword_index != -1:  # 如果找到关键词
                if first_keyword is None or keyword_index < first_keyword[1]:
                  first_keyword = (keyword, keyword_index)
                # 将 st.session_state.M 设置为找到的第一个关键词
            if first_keyword:
              st.session_state.M = first_keyword[0]
        
        if "0" in Action :
          with st.spinner("Executing Action0..."):
            answer_list1 = ast.literal_eval(decision)
            st.session_state.Uin, st.session_state.Uo,st.session_state.P = answer_list1   
            Current_Stress,nZVS,nZCS,P,pos,plot,M=test3.PINN(st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.M)
            Current_Stress1,nZVS1,nZCS1,P1,pos1,plot1,M1=test3.PINN(st.session_state.Uin,st.session_state.Uo,100,st.session_state.M)
            Current_Stress_sps,nZVS_sps,nZCS_sps,P_sps, pos_sps, plot_sps, M_sps=test3.PINN(st.session_state.Uin,st.session_state.Uo,st.session_state.P,"SPS")
            Current_Stress_sps1,nZVS_sps1,nZCS_sps1,P_sps1, pos_sps1, plot_sps1, M_sps1=test3.PINN(st.session_state.Uin,st.session_state.Uo,100,"SPS")
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
        elif "1" in Action:
          with st.spinner("Executing Action1..."): 
            Current_Stress,nZVS,nZCS,P,pos,plot,M=test3.PINN(st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.M)
            Current_Stress1,nZVS1,nZCS1,P1,pos1,plot1,M1=test3.PINN(st.session_state.Uin,st.session_state.Uo,100,st.session_state.M)
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
        elif "2" in Action:
          with st.spinner("Executing Action2..."):
            answer_list = ast.literal_eval(decision)
            st.session_state.v_ripple_lim, st.session_state.i_ripple_lim,st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.fs = answer_list
            st.session_state.L,st.session_state.C,st.session_state.i_ripple_value,st.session_state.v_ripple_value,st.session_state.i_ripple_percentage,st.session_state.v_ripple_percentage ,st.session_state.iLdc,st.session_state.iL1,st.session_state.iL2,st.session_state.iL3,st.session_state.Vodc,st.session_state.Vo1,st.session_state.Vo2,st.session_state.Vo3,st.session_state.P_on,st.session_state.P_off,st.session_state.P_cond=test_buck.optimization(st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.fs,st.session_state.i_ripple_lim, st.session_state.v_ripple_lim)
            Answer1=test_buck.answer1(st.session_state.L,st.session_state.C,st.session_state.v_ripple_value,st.session_state.v_ripple_percentage,st.session_state.i_ripple_value,st.session_state.i_ripple_percentage)
            Answer2="The output waveform of the inductor current in steady state under this operating condition is shown in the following figure:"
            Answer3="The output waveform of the output voltage in steady state under this operating condition is shown in the following figure:"
            plot1,plot2=test_buck.draw(st.session_state.L,st.session_state.C,st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.fs)
            st.write(Answer1)
            st.write(Answer2)
            st.image(plot1)    
            st.write(Answer3)
            st.image(plot2)
            message = {"role": "assistant", "content": Answer1}
            message1 = {"role": "assistant", "content": Answer2,"images": plot1}
            message2 = {"role": "assistant", "content": Answer3,"images": plot2}
            st.session_state.messages.append(message)
            st.session_state.messages.append(message1)
            st.session_state.messages.append(message2)
        elif "3" in Action:
          with st.spinner("Executing Action3..."):
            reply=test_buck.answer2(st.session_state.iLdc,st.session_state.iL1,st.session_state.iL2,st.session_state.iL3,st.session_state.Vodc,st.session_state.Vo1,st.session_state.Vo2,st.session_state.Vo3)
            st.write(reply)
            message = {"role": "assistant", "content": reply}
            st.session_state.messages.append(message)
        elif "4" in Action:
          with st.spinner("Executing Action4..."):
            with st.spinner("Waiting... PLECS is starting up..."):
              Buck_plecs.startplecs(st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.L,st.session_state.C,st.session_state.fs)
              reply="The PLECS simulation is running… Complete! You can now verify if the design is reasonable by observing the simulation waveforms."
              st.write(reply)
              message = {"role": "assistant", "content": reply}
              st.session_state.messages.append(message)
        elif "5" in Action:
          with st.spinner("Executing Action5..."):
            st.write(decision)
            message = {"role": "assistant", "content": decision}
            st.session_state.messages.append(message)
        else:
          with st.spinner("Executing Action6..."):
            st.write(decision)
            message = {"role": "assistant", "content": decision}
            st.session_state.messages.append(message)
            




    # elif "ripple constraint" in prompt.lower() and "operating conditions" in prompt.lower():
    #     with st.chat_message("assistant"):
    #          with st.spinner("Well received. Please hold on for a while. Analysing….. "):
    #             response = chat_engine.chat(prompt, messages_history)
    #             answer_list = ast.literal_eval(response.response)
    #             st.session_state.v_ripple_lim, st.session_state.i_ripple_lim,st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.fs = answer_list
    #             st.session_state.L,st.session_state.C,st.session_state.i_ripple_value,st.session_state.v_ripple_value,st.session_state.i_ripple_percentage,st.session_state.v_ripple_percentage ,st.session_state.iLdc,st.session_state.iL1,st.session_state.iL2,st.session_state.iL3,st.session_state.Vodc,st.session_state.Vo1,st.session_state.Vo2,st.session_state.Vo3,st.session_state.P_on,st.session_state.P_off,st.session_state.P_cond=test_buck.optimization(st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.fs,st.session_state.i_ripple_lim, st.session_state.v_ripple_lim)
    #             Answer1=test_buck.answer1(st.session_state.L,st.session_state.C,st.session_state.v_ripple_value,st.session_state.v_ripple_percentage,st.session_state.i_ripple_value,st.session_state.i_ripple_percentage)
    #             Answer2="The output waveform of the inductor current in steady state under this operating condition is shown in the following figure:"
    #             Answer3="The output waveform of the output voltage in steady state under this operating condition is shown in the following figure:"
    #             plot1,plot2=test_buck.draw(st.session_state.L,st.session_state.C,st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.fs)
    #             st.write(Answer1)
    #             st.write(Answer2)
    #             st.image(plot1)    
    #             st.write(Answer3)
    #             st.image(plot2)
    #             message = {"role": "assistant", "content": Answer1}
    #             message1 = {"role": "assistant", "content": Answer2,"images": plot1}
    #             message2 = {"role": "assistant", "content": Answer3,"images": plot2}
    #             st.session_state.messages.append(message)
    #             st.session_state.messages.append(message1)
    #             st.session_state.messages.append(message2)
    # elif "harmonic components" in prompt.lower():
    #     with st.chat_message("assistant"):
    #          with st.spinner("Thinking..."):

    # elif "C2M0080120D" in prompt:
    #     with st.chat_message("assistant"):    
    #           with st.spinner("Calculating"):
    #             reply=test_buck.answer3(st.session_state.P_on,st.session_state.P_off,st.session_state.P_cond)
    #             st.write(reply)
    #             message = {"role": "assistant", "content": reply}
    #             st.session_state.messages.append(message)
    # elif "verify" in prompt.lower():
    #     with st.chat_message("assistant"):    
    #           with st.spinner("Waiting... PLECS is starting up..."):
    #             Buck_plecs.startplecs(st.session_state.Uin,st.session_state.Uo,st.session_state.P,st.session_state.L,st.session_state.C,st.session_state.fs)
    #             reply="The PLECS simulation is running… Complete! You can now verify if the design is reasonable by observing the simulation waveforms."
    #             st.write(reply)
    #             message = {"role": "assistant", "content": reply}
    #             st.session_state.messages.append(message)
    # else:
    #      with st.chat_message("assistant"):
    #          with st.spinner("Loading..."):
    #             #response = chat_engine.chat(prompt, messages_history)
    #             response=client.chat.completions.create(
    #                     model=st.session_state["openai_model"],
    #                 messages=[
    #                     {"role":"system","content":"You are now an expert in the power electronics industry, and you are proficient in optimal design of buck converter.Please answer the questions  in a warm, positive and friendly manner.Keep your answer less than 150 words! Make sure your answers are professional and accurate -- don't hallucinate."},
    #                     *[
    #                         {"role": m["role"], "content": m["content"]}
    #                     for m in st.session_state.messages
    #                       ]
    #                 ],

    #                      stream=False,
    #             )
    #             st.write(response.choices[0].message.content)
    #             message = {"role": "assistant", "content": response.choices[0].message.content}
    #             st.session_state.messages.append(message)

    #             # response = chat_engine.chat(prompt, messages_history)
    #             # st.write(response.response)
    #             # message = {"role": "assistant", "content": response.response}
    #             # st.session_state.messages.append(message)
