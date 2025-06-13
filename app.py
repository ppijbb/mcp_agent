import streamlit as st
import requests
import json

st.title("vLLM Chat Completion Test")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare request to vLLM server
    vllm_url = "http://34.47.83.72/llmservice/v1/generate/chat/completions"
    print(st.session_state.messages)
    try:
        with requests.post(
            vllm_url,
            json={
                "model": "your-model-name",
                "messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": True
            },
            stream=True
        ) as response:
            if response.status_code == 200:
                # Create a placeholder for the assistant's response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Process the streaming response
                    for line in response.iter_lines():
                        if line:
                            try:
                                full_response += line.decode('utf-8')
                                message_placeholder.markdown(full_response + "▌")
                            except json.JSONDecodeError:
                                continue
                    
                    # Update the final response without the cursor
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        st.error(f"Error connecting to vLLM server: {str(e)}") 