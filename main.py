import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory, CombinedMemory, VectorStoreRetrieverMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from profileloader import load_user_profile
from profileupdater import get_profile_update


# ----------------- SETUP -----------------
load_dotenv()
def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return '{"update": "None"}'

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("Missing MONGO_URI in .env")

client = MongoClient(MONGO_URI)
db = client["elder_companion"]

USER_ID = "elder_001"


profile_context = load_user_profile(USER_ID)


# Embeddings + LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

# Vector store (shared collection, filtered by user_id)
vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    f"{db.name}.long_term_memory",
    embeddings,
    index_name="elder_memory_index"  # must exist in MongoDB Atlas
)

# Retriever memory with filter for user_id
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5},
    search_filter={"user_id": USER_ID}
)
long_term_memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="relevant_context",
    input_key="input"
)

# Short-term memory
short_term_memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="history",
    input_key="input",
    return_messages=True
)

# Hybrid memory
hybrid_memory = CombinedMemory(memories=[short_term_memory, long_term_memory])

# ----------------- PROMPTS -----------------
conversation_prompt = PromptTemplate(
    input_variables=["history", "relevant_context", "input"],
    template=f"""
The following is a conversation between a Human and an empathetic AI companion.
Use profile context, long-term memories (summaries), and recent history to respond naturally.
Suggest activities that the user might enjoy based on their preferences and mood.

--- PROFILE CONTEXT ---
{profile_context}

--- SUMMARY MEMORIES ---
{{relevant_context}}

--- RECENT HISTORY ---
{{history}}

Human: {{input}}  
AI:"""
)

conversation = LLMChain(
    llm=llm,
    memory=hybrid_memory,
    prompt=conversation_prompt,
    verbose=True
)

# ----------------- MEMORY EXTRACTION -----------------
extraction_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
extraction_prompt = PromptTemplate(
    input_variables=["input", "response"],
    template="""
You are a memory extraction agent.
From this conversation, extract:
 SUMMARY (short sentence for semantic memory)

Conversation:
Human: {input}
AI: {response}

Output in JSON:
{{
  "summary": "The user mentioned his son Arjun who committed suicide."
}}

If no new memory: {{ "summary": "None" }}
"""
)

memory_extraction_chain = LLMChain(llm=extraction_llm, prompt=extraction_prompt)


# ----------------- CHAT LOOP -----------------
print("🤖 Elder Companion ready. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye! Take care ❤️")
        break

    # Get AI response
    ai_response = conversation.predict(input=user_input)
    print(f"AI: {ai_response}")

    # Extract memories
    extracted = memory_extraction_chain.run(input=user_input, response=ai_response)
    profile_update = get_profile_update(user_input,ai_response)
    print(profile_update)
    

    try:
        extracted_json = json.loads(extracted)
        #update_json = json.loads(profile_update)
        #print("all ok")
        
    except:
        extracted_json = {"summary": "None"}

    # --- Save summaries in new schema ---
    if extracted_json.get("summary") and extracted_json["summary"].lower() != "none":
        vectorstore.add_texts(
            [extracted_json["summary"]],
            metadatas=[{
                "user_id": USER_ID,
                "type": "summary",
                "category": "conversation",
                "source": "chat",
                "timestamp": datetime.utcnow().isoformat()
            }]
        )
        print(f"Memory saved ✅: {extracted_json['summary']}")

    # --- Update profile favorites ---
    try:
        profile_update_json_str = extract_json(profile_update)
        update_json = json.loads(profile_update_json_str)
        #update_json = json.loads(profile_update)
        #print(update_json)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        print("herer ")
        #update_json = {"update": "None"}

    if update_json.get("update") != "None":
        update_fields = {}
        for key in ["songs", "movies", "books", "photos", "hobbies"]:
            if key in update_json and update_json[key]:
                update_fields.setdefault(f"{key}", {"$each": update_json[key]})

        if update_fields:
            db.long_term_memory.update_one(
                {"user_id": USER_ID, "type": "profile"},
                {"$addToSet": update_fields},  # prevents duplicates
                upsert=True
            )
            print("✅ Profile updated with new favorites")

    print("-" * 50)
