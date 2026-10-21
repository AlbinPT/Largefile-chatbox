import requests, streamlit as st

st.title("Medication Safety Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

if question := st.chat_input("Ask something..."):
    st.chat_message("user").write(question)
    with st.spinner("Thinking..."):
        r = requests.post("http://localhost:8000/chat", json={"question": question}).json()
    st.chat_message("assistant").write(r["answer"])
    with st.expander("Sources"):
        for s in r["sources"]:
            st.write(f'{s["file"]} (page {s["page"]})')
    st.session_state.history += [("user", question), ("assistant", r["answer"])]