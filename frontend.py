import streamlit as st
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr
import requests
import uuid

st.title("Chat with mAIstro")

# --- Login Functionality ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:
    st.header("Login")
    username = st.text_input("Enter your username")
    if st.button("Login"):
        if username:
            st.session_state.user_id = username
            st.session_state.threads = {}  # {thread_id: messages}
            st.session_state.active_thread = None
            st.rerun()
        else:
            st.error("Please enter a username")
else:
    # --- Sidebar for Threads ---
    st.sidebar.header(f"Welcome, {st.session_state.user_id}")
    if st.sidebar.button("New Chat"):
        thread_id = str(uuid.uuid4())
        st.session_state.threads[thread_id] = []
        st.session_state.active_thread = thread_id

    st.sidebar.header("Chat Threads")
    for thread_id in st.session_state.threads:
        if st.sidebar.button(f"Chat {thread_id[:8]}"):
            st.session_state.active_thread = thread_id

    # --- Chat Interface ---
    if st.session_state.active_thread:
        st.header(f"Thread: {st.session_state.active_thread[:8]}")
        
        # Display existing messages for the active thread
        for message in st.session_state.threads[st.session_state.active_thread]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Create columns for text input and voice button
        col1, col2 = st.columns([5, 1])
        
        with col1:
            prompt = st.chat_input("What is up?")
        
        with col2:
            if st.button("🎙️", key="voice_button", help="Click to record voice"):
                r = sr.Recognizer()
                try:
                    with sr.Microphone() as source:
                        st.info("Please speak now...")
                        audio = r.listen(source, timeout=10)
                    
                    voice_text = r.recognize_google(audio)
                    st.success(f"Recognized: {voice_text}")
                    prompt = voice_text
                    
                except sr.UnknownValueError:
                    st.error("Could not understand audio. Please try again.")
                except sr.RequestError as e:
                    st.error(f"Error with speech recognition: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Handle prompt (from both text and voice input)
        if prompt:
            # Add user message to the active thread and display it
            st.session_state.threads[st.session_state.active_thread].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Prepare the request payload
            payload = {
                "thread_id": st.session_state.active_thread,
                "user_id": st.session_state.user_id,
                "messages": [{"role": "user", "content": prompt}]
            }

            # Send request to the FastAPI backend
            try:
                response = requests.post("http://localhost:8000/chat", json=payload)
                response.raise_for_status()
                
                response_data = response.json()
                ai_response = response_data.get("response", "Sorry, I couldn't get a response.")

                # Add AI response to the active thread and display it
                st.session_state.threads[st.session_state.active_thread].append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                
                # Rerun to update the message display
                st.rerun()

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
    else:
        st.info("Select a chat thread or start a new one from the sidebar.")
