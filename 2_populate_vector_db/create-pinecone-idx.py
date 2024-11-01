# create pinecone db index
import os
from pinecone import Pinecone, ServerlessSpec

PINECONE_INDEX = os.getenv('PINECONE_INDEX')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_REGION = os.getenv('PINECONE_REGION')

pc = Pinecone(api_key=PINECONE_API_KEY)
print("Pinecone initialised")

print(f"Creating 768-dimensional index called '{PINECONE_INDEX}'...")
# Create the Pinecone index with the specified dimension.
pc.create_index(name=PINECONE_INDEX,
                dimension=768,
                spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_REGION))
print("Success")