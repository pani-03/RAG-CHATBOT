import streamlit as st
import os
import uuid
from backend import chatbot, retrieve_all_threads, pdf_text_extractor
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# ---- Session State Setup ----
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ---- Sidebar ----
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

st.sidebar.markdown("---")
lang = st.sidebar.selectbox("Chatbot Language", ["English", "Hindi", "French", "German"])
st.session_state["chatbot_lang"] = lang

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    temp_path = f"temp_{st.session_state['thread_id']}.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["uploaded_file_path"] = temp_path

if st.sidebar.button("Export Chat History"):
    chat_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["message_history"]])
    st.download_button('Download Chat', chat_str, file_name='chat_history.txt')

# ---- Main Chat Area ----
st.title("LangGraph Chatbot Demo 🚀")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])  # markdown for better formatting

user_input = st.chat_input("Type here")

if user_input or "uploaded_file_path" in st.session_state:
    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        input_list = [HumanMessage(content=user_input)]
    else:
        # ---- Handle PDF Upload ----
        pdf_result = pdf_text_extractor(st.session_state['uploaded_file_path'])
        if "error" in pdf_result:
            st.session_state["message_history"].append(
                {"role": "assistant", "content": f"❌ PDF Error: {pdf_result['error']}"}
            )
            st.stop()
        else:
            pdf_text = pdf_result["text"]
            st.session_state["message_history"].append(
                {"role": "user", "content": f"📄 Uploaded PDF Content:\n\n{pdf_text[:1000]}..."}
            )
            with st.chat_message("user"):
                st.markdown(f"📄 Uploaded PDF content:\n\n{pdf_text[:1000]}...")
            input_list = [HumanMessage(content=pdf_text)]

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"], "lang": st.session_state["chatbot_lang"]},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": input_list},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
