# ToDo mAIstro: A Personalized AI To-Do List Manager

**ToDo mAIstro** is a smart chatbot assistant designed to help you manage your tasks and personal information. It uses a sophisticated AI backend to understand your conversations, automatically update your ToDo list, and remember your personal details (like your name, location, and interests) to provide a truly personalized experience.

The entire system is built with a separate Streamlit frontend and FastAPI backend, featuring speech-to-text input and end-to-end AI observability with Langfuse.

## ✨ Features

* **Personalized Memory:** Remembers your user profile (name, job, etc.), a detailed ToDo list, and custom preferences for how you like your tasks managed.
* **Automatic Updates:** Uses **Trustcall** to intelligently and reliably extract information from your conversation and update your long-term memory.
* **Multi-Threaded Chat:** Manage multiple, separate chat sessions, each with its own context.
* **User Authentication:** A simple username-based login system keeps memories and threads separate for each user.
* **Voice Input:** Speak your commands directly to the app using the built-in speech-to-text recognition.
* **AI Observability:** Fully integrated with **Langfuse** to trace and debug every step of the AI's reasoning and memory updates.
* **Scalable Architecture:** Decoupled frontend and backend allows for independent scaling and development.

## 🏗️ Architecture

The application operates on a client-server model:

1.  **Frontend (`frontend.py`):** A **Streamlit** application provides the user interface. It handles user login, thread management, and captures both text and voice input (using `streamlit-webrtc` and `SpeechRecognition`). It then sends user messages to the FastAPI backend.

2.  **Backend (`api.py`):** A **FastAPI** server exposes the `/chat` and `/memory/{user_id}` endpoints.

3.  **AI Core (`api.py`):** The backend uses **LangGraph** to create a stateful graph (a state machine). This graph:
    * Receives the user's message.
    * Loads the user's current `Profile`, `ToDo` list, and `Instructions` from memory.
    * Uses a Google Gemini model (`gemini-2.5-pro`) to decide if a memory update is needed.
    * Routes to specific nodes (`update_profile`, `update_todos`, `update_instructions`) if an update is required.
    * Uses **Trustcall** to perform structured, reliable updates to the memory schemas.
    * Generates a natural response for the user.

4.  **Memory:** The `InMemoryStore` from LangGraph is used to persist user data (profile, todos, instructions) across different sessions, namespaced by `user_id`.



## 🛠️ Tech Stack

* **Backend:** FastAPI, LangGraph, Trustcall
* **Frontend:** Streamlit, Streamlit-WebRTC
* **LLM:** Google Gemini (via `langchain_google_genai`)
* **Observability:** Langfuse
* **Voice:** SpeechRecognition, PyAudio, pydub, gTTS
* **Dependencies:** Pydantic, python-dotenv

---

## 🚀 Setup and Installation

Follow these steps to get the application running locally.

### 1. Prerequisites

* Python 3.9+
* A Google API Key (for the Gemini model)
* A Langfuse account (for observability)

### 2. Clone the Repository

```bash
git clone 
replace your environment variables
run bat file 