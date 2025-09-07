import os
from dotenv import load_dotenv
from pymongo import MongoClient

# ----------------- PROFILE MEMORY -----------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("Missing MONGO_URI in .env")

client = MongoClient(MONGO_URI)
db = client["elder_companion"]

# Same collection where memories & profiles are stored
vector_collection = db["long_term_memory"]

def load_user_profile(user_id="elder_001"):
    profile = vector_collection.find_one(
        {"user_id": user_id, "type": "profile"},
        {"_id": 0}
    )

    if not profile:
        return "No profile information available."
    
    profile_text = []
    if profile.get("songs"):
        profile_text.append(f"Favourite songs: {', '.join(profile['songs'])}")
    if profile.get("movies"):
        profile_text.append(f"Favourite movies: {', '.join(profile['movies'])}")
    if profile.get("books"):
        profile_text.append(f"Favourite books: {', '.join(profile['books'])}")
    if profile.get("photos"):
        profile_text.append(f"Photo memories: {', '.join(profile['photos'])}")
    if profile.get("hobbies"):
        profile_text.append(f"Hobbies: {', '.join(profile['hobbies'])}")

    return "\n".join(profile_text)

# Example usage

