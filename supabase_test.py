from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

# example data
fake_data = {
    "completed_course": [480, 580, 599, 601],
    "gradTarget": "Spring 2026"
}

insert_response = supabase.table("Example Table").insert(fake_data).execute()

print("Inserted:")
print(insert_response.data)

response = supabase.table("Example Table") \
    .select("completed_course") \
    .eq("uid", 4) \
    .execute()

print(response.data)
