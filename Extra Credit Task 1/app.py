import streamlit as st
import requests
import uuid

# --- Configuration ---
BASE_API_URL = "http://localhost:7860"
FLOW_ID = "c7c74e1c-d129-4f8c-a210-b0d02e5731a5"
CHAT_INPUT_COMPONENT_ID = "ChatInput-JAdwX"
url = f"{BASE_API_URL}/api/v1/run/{FLOW_ID}"

APPLICATION_TOKEN = "sk-OJo6bYQ47WGxbeoDj2hsmo-5_Y7AJarWGqaAhQ0UlA8"

def upload_file_to_langflow(file_bytes, file_name: str) -> str:
    """
    Uploads a file to the Langflow server and returns the server's file path.
    """
    api_url = f"{BASE_API_URL}/api/v1/files/upload/{FLOW_ID}"
    files = {"file": (file_name, file_bytes)}
    headers = {}
    
    try:
        # Requests automatically handles the multipart/form-data boundary when using 'files'
        response = requests.post(api_url, files=files, headers=headers)
        response.raise_for_status()
        return response.json().get("file_path")
    except requests.exceptions.HTTPError as e:
        # Fallback to authentication if unauthorized
        headers["x-api-key"] = APPLICATION_TOKEN
        response = requests.post(api_url, files=files, headers=headers)
        response.raise_for_status()
        return response.json().get("file_path")

def run_flow(message: str, session_id: str, file_path: str = None) -> dict:
    """
    Sends the chat message (and optional file tweak) to the Langflow API.
    """
    tweaks = {
        CHAT_INPUT_COMPONENT_ID: {"input_value": message},
    }
    if file_path:
        tweaks["File"] = {"path": file_path}

    payload = {
        "output_type": "chat",
        "input_type": "chat",
        "tweaks": tweaks,
        "session_id": session_id,
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": APPLICATION_TOKEN,
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def extract_response_text(response: dict) -> str:
    """
    Extracts the text from the nested Langflow response JSON.
    """
    try:
        return response["outputs"][0]["outputs"][0]["results"]["message"]["text"]
    except (KeyError, IndexError, TypeError):
        try:
            return response["outputs"][0]["outputs"][0]["artifacts"]["message"]
        except (KeyError, IndexError, TypeError):
            try:
                return response["result"]["message"]["text"]
            except (KeyError, TypeError):
                return f"Received response but couldn't parse. Raw: {response}"

# --- Streamlit UI ---
st.title("Langflow RAG Chatbot")
st.markdown("Powered by Vertex AI & Chroma DB")

# Sidebar for Document Upload
with st.sidebar:
    st.header("Document Upload")
    st.markdown("Upload a file to act as the context for the RAG chatbot.")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx", "csv"])
    
    if uploaded_file is not None:
        # Only upload if it's a newly uploaded file
        if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            with st.spinner("Uploading file to Langflow..."):
                try:
                    file_bytes = uploaded_file.read()
                    file_path = upload_file_to_langflow(file_bytes, uploaded_file.name)
                    st.session_state.uploaded_file_path = file_path
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.success(f"File '{uploaded_file.name}' uploaded and ready!")
                except Exception as e:
                    st.error(f"Failed to upload file: {e}")
        else:
            st.success(f"File '{uploaded_file.name}' is loaded.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Display previous chat messages on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the document or policy..."):
    # 1. Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # 2. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Fetch and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Retrieve uploaded file path if one exists
                current_file_path = st.session_state.get("uploaded_file_path")
                
                # Call Langflow API
                raw_response = run_flow(
                    prompt,
                    session_id=st.session_state.session_id,
                    file_path=current_file_path,
                )
                
                # Parse the output
                response_text = extract_response_text(raw_response)
                
                # Display response
                st.markdown(response_text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Is your Langflow server running on port 7860?")
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"An error occurred: {e}")