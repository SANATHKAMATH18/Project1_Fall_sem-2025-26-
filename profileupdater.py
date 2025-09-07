
import os
import json
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI


def get_profile_update(i,r):
    """
    Creates and returns an LLMChain for updating user profile information 
    (songs, movies, books, photos, hobbies) from a conversation.
    """
    # Load env vars
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env")

    # Define prompt template
    profile_update_prompt = PromptTemplate(
        input_variables=["input", "response"],
        template="""
You are a profile update agent.
From the conversation, identify if the user mentioned a favorite or liked in songs, movies, books, photos, or hobbies. 
Output ONLY in JSON with arrays for the categories. 
If nothing new, return {{"update": "None"}}.

Conversation:
Human: {input}
AI: {response}

Example Output:
{{
  "songs": ["Ae Mere Watan Ke Logon"],
  "movies": [],
  "books": ["The Discovery of India"],
  "photos": [],
  "hobbies": ["painting"]
}}
"""
    )

    # Create LLM and chain
    profile_update_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=google_api_key
    )
    profile_update_chain = LLMChain(llm=profile_update_llm, prompt=profile_update_prompt)
# Inside chat loop, after ai_response
    profile_update = profile_update_chain.run(input=i, response=r)
    return profile_update


print(get_profile_update("I love the song Imagine by John Lennon.","That's a great song! It really promotes peace and unity."))








