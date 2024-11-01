import gradio as gr
import cmlapi
import pinecone
from pinecone import Pinecone, ServerlessSpec
from typing import Any, Union, Optional
from pydantic import BaseModel
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import requests
import json
import time
from typing import Optional
import boto3
from botocore.config import Config
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import hf_hub_download
import shutil
import os

# fixed duplicate pdf upload docs


# Set any of these to False, if not using respective parts of the lab
USE_PINECONE = True 
USE_CHROMA = True 

#EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
#project_name = 'BBH-LLM-POV'

    
## TO DO GET MODEL DEPLOYMENT
## Need to get the below prgramatically in the future iterations
client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
projects = client.list_projects(include_public_projects=True, search_filter=json.dumps({"name": "Shared LLM Model for Hands on Lab"}))
project = projects.projects[0]

## Here we assume that only one model has been deployed in the project, if this is not true this should be adjusted (this is reflected by the placeholder 0 in the array)
model = client.list_models(project_id=project.id)
selected_model = model.models[0]

## Save the access key for the model to the environment variable of this project
MODEL_ACCESS_KEY = selected_model.access_key

MODEL_ENDPOINT = os.getenv("CDSW_API_URL").replace("https://", "https://modelservice.").replace("/api/v1", "/model?accessKey=")
MODEL_ENDPOINT = MODEL_ENDPOINT + MODEL_ACCESS_KEY

## set up for running jobs via api

## To-do add a dynamic project discover step
## what project is this search
project_name = 'BBH-LLM-DEMO-EMBD'

#proj_id = project.id
proj_id = client.list_projects(search_filter=json.dumps({"name":project_name })).projects[0].id

print("project_id",proj_id)

if os.environ.get("AWS_DEFAULT_REGION") == "":
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

    
## Setup Bedrock client:
def get_bedrock_client(
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    endpoint_url :
        Optional override for the Bedrock service API Endpoint. If setting this, it should usually
        include the protocol i.e. "https://..."
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)


    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    bedrock_client = session.client(
        service_name="bedrock-runtime",
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client


boto3_bedrock = get_bedrock_client(
      region=os.environ.get("AWS_DEFAULT_REGION", None))


def main():
    # Configure gradio QA app 
    print("Configuring gradio app")

    DESC = "This AI-powered assistant showcases the flexibility of Cloudera Machine Learning to work with 3rd party solutions for LLMs and Vector Databases, as well as internally hosted models and vector DBs. The prototype does not yet implement chat history and session context - every prompt is treated as a brand new one."

    # Create the Gradio Interface using Blocks
    with gr.Blocks() as demo:
        with gr.Tab("Chat Interface"):
            gr.Markdown(DESC)

            # Components
            chatbot = gr.Chatbot()
            message_input = gr.Textbox(show_label=False, placeholder="Type your message here")

            # Additional inputs - make sure they are added to the UI
            with gr.Row():
                model_choice = gr.Radio(
                    ['Local Mistral 7B', 'AWS Bedrock Claude v2.1'],
                    label="Select Foundational Model",
                    value="Local Mistral 7B"
                )
                temperature = gr.Slider(
                    minimum=0.01, maximum=1.0, step=0.01, value=0.5,
                    label="Select Temperature (Randomness of Response)"
                )
            with gr.Row():
                tokens = gr.Radio(
                    ["50", "100", "250", "500", "1000"],
                    label="Select Number of Tokens (Length of Response)",
                    value="250"
                )
                vector_db_choice = gr.Radio(
                    ['None', 'Pinecone', 'Chroma'],
                    label="Vector Database Choices",
                    value="Pinecone"
                )
                embedding_model = gr.Radio(
                    ['all-mpnet-base-v2','LaBSE','facebook-dpr-ctx_encoder-multiset-base',
'msmarco-bert-base-dot-v5'],
                    label="Embedding Model Choices - pdf upload must match query model",
                    value='all-mpnet-base-v2'
                )

            send_button = gr.Button("Send")

            # Collapsible source box
            with gr.Accordion("Source Information (click to expand)", open=False):
                source_output = gr.Textbox(label="Source", interactive=False)

            # Function to handle user input and update outputs
            def submit_message(message, history, model_choice_value, temperature_value, tokens_value, vector_db_choice_value, embedding_model_value):
                global EMBEDDING_MODEL_NAME
                global EMBEDDING_MODEL_REPO
                
                EMBEDDING_MODEL_NAME = embedding_model_value
                EMBEDDING_MODEL_REPO = f"sentence-transformers/{embedding_model_value}"
                
                # define environment variable each time
                os.environ['EMBEDDING_MODEL_REPO'] = EMBEDDING_MODEL_REPO   
                
                gen = get_responses(
                    message, history, model_choice_value, temperature_value,
                    tokens_value, vector_db_choice_value
                )
            
                for new_history, context_chunk in gen:
                    # If context_chunk is None, we use gr.update() to indicate no change
                    if context_chunk is None:
                        yield new_history, gr.update()
                    else:
                        # When context_chunk is available, we update the source_output
                        print('within submit msg, context_chunk is:',context_chunk[0:10])
                        yield new_history, context_chunk

            # Connect the function to the UI components
            send_button.click(
                fn=submit_message,
                inputs=[message_input, chatbot, model_choice, temperature, tokens, vector_db_choice,embedding_model],
                outputs=[chatbot, source_output]
            )

         
        
        with gr.Tab("Upload PDF"):
            # Function to handle PDF upload and set EMBEDDING_MODEL_REPO
            def handle_pdf_upload_with_model(pdf_files, embedding_model_value):
                global EMBEDDING_MODEL_REPO
                
                # Set EMBEDDING_MODEL_REPO based on the embedding_model_value
                EMBEDDING_MODEL_REPO = f"sentence-transformers/{embedding_model_value}"
                
                print("EMBEDDING_MODEL_REPO",EMBEDDING_MODEL_REPO)
                
                # Set the environment variable
                os.environ['EMBEDDING_MODEL_REPO'] = EMBEDDING_MODEL_REPO

                # Call the original PDF handling function
                return handle_pdf_upload(pdf_files)

            # Retain the PDF upload functionality and connect embedding model
            upload_demo = gr.Interface(
                fn=handle_pdf_upload_with_model,
                inputs=[
                    gr.File(
                        label="Upload PDF(s)", 
                        file_count="multiple",  # This is the key change - more reliable than multiselect=True
                        file_types=[".pdf"]
                    ),
                    embedding_model
                ],
                outputs="text",
                allow_flagging="never"
            )
            #upload_demo.render() this was creating a duplicate upload 

    print("Launching Gradio app")

    demo.launch(
        share=True,
        enable_queue=True,
        show_error=True,
        # Adjust server_name and server_port as needed
        server_name='127.0.0.1',
        server_port=int(os.getenv('CDSW_READONLY_PORT', '7860'))
    )

    print("Gradio app ready")

    

# if USE_PINECONE:
#     PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
#     PINECONE_INDEX = os.getenv('PINECONE_INDEX')

#     print("initialising Pinecone connection...")
#     pc = Pinecone(api_key=PINECONE_API_KEY)
#     print("Pinecone initialised")

#     print(f"Getting '{PINECONE_INDEX}' as object...")
#     index = pc.Index(PINECONE_INDEX)
#     print("Success")

#     # Get latest statistics from index
#     current_collection_stats = index.describe_index_stats()
#     print('Total number of embeddings in Pinecone index is {}.'.format(current_collection_stats.get('total_vector_count')))

    
#if USE_CHROMA:

        
    
# Function to handle PDF uploads
def handle_pdf_upload(pdf_files):
    if pdf_files is not None:
        staged_files_dir = '/home/cdsw/staged_files'
        print('desired file path is', staged_files_dir)
        print('type of pdf_files', type(pdf_files))
        
        # Initialize a list to store the results
        result_messages = ["Processing uploaded files... Please wait while the jobs are running."]

        # Handle both single file and multiple files
        if isinstance(pdf_files, list):
            files_to_process = pdf_files
        else:
            files_to_process = [pdf_files]

        # Iterate over the list of uploaded files
        for pdf_file in files_to_process:
            # Extract just the file name (without the temp directory path)
            file_name = os.path.basename(pdf_file.name)
            file_path = os.path.join(staged_files_dir, file_name)
            print('current file is', file_name)
            
            # Move each file to the 'staged_files' directory
            shutil.move(pdf_file.name, file_path)

            # Print the final destination path for debugging
            print(f"File moved to: {file_path}")
            
            # Add a success message for each file
            result_messages.append(f"File {file_name} has been uploaded and moved to 'staged_files'.")
        # let user know staring job chain
        result_messages.append("Starting job chain...")
        # Kick off the job chain after the upload
        job_chain = ['pdf_reader', 'Populate Chroma Vector DB', 'populate pinecone']
        
        for job in job_chain:
            print(f'running {job}')
            if job == 'pdf_reader':
                target_job = client.list_jobs(proj_id, search_filter=json.dumps({"name": job}))
                print('job running is', job)
                #print('target_job is',target_job)
                print('target_job id',target_job.jobs[0].id)
                #print('type of target_job.jobs',type(target_job.jobs))
                job_run = client.create_job_run(cmlapi.CreateJobRunRequest(),project_id = proj_id, job_id = target_job.jobs[0].id)
                job_status = ''
              
                while job_status != 'ENGINE_SUCCEEDED':
                    # api forcing 50 job runs to show up per page. 
                    # after 50 job runs, this will not work
                    print('checking job status')
                    listed_jobs = client.list_job_runs(project_id=proj_id, job_id=target_job.jobs[0].id, page_size=50)
                    job_status = listed_jobs.job_runs[-1].status

                    if job_status == 'ENGINE_FAILED':
                        print(f'{job} has failed.')
                        raise RuntimeError(f"Job {job} failed. Halting the job chain.")
                        
                    # Update the user that the job is still processing
                    result_messages.append(f"{job} is still processing... Please wait.")
                    time.sleep(5)
            else:
                target_job = client.list_jobs(proj_id, search_filter=json.dumps({"name": job}))
                job_run = client.create_job_run(cmlapi.CreateJobRunRequest(),project_id = proj_id, job_id = target_job.jobs[0].id)

            print(f'{job} is completed')
            result_messages.append(f"All jobs completed")

        # Return the result messages as a single string
        return "\n".join(result_messages)
    
    return "No files uploaded."

    
# Helper function for generating responses for the QA app
def get_responses(message, history, model, temperature, token_count, vector_db):
    if history is None:
        history = []
    if model == "Local Mistral 7B":
        if vector_db == "None":
            context_chunk = ""
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
            bot_message = ""
            for i in range(len(response)):
                time.sleep(0.02)
                bot_message = response[:i+1]
                yield history + [(message, bot_message)], None  # Yield None for context_chunk

        elif vector_db == "Pinecone":
            
            PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
            PINECONE_INDEX = os.getenv('PINECONE_INDEX')

            print("initialising Pinecone connection...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            print("Pinecone initialised")

            print(f"Getting '{PINECONE_INDEX}' as object...")
            index = pc.Index(PINECONE_INDEX)
            print("Success")

            # Get latest statistics from index
            current_collection_stats = index.describe_index_stats()
            print('Total number of embeddings in Pinecone index is {}.'.format(current_collection_stats.get('total_vector_count')))

            
            context_chunk, source, score = get_nearest_chunk_from_pinecone_vectordb(index, message)
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
            response = f"{response}\n\n For additional info see source information below"
            bot_message = ""
            for i in range(len(response)):
                time.sleep(0.02)
                bot_message = response[:i+1]
                if i == len(response) - 1:
                    print('string is',context_chunk[:4])
                    yield history + [(message, bot_message)], context_chunk
                else:
                    yield history + [(message, bot_message)], None
        elif vector_db == "Chroma":
            
                # Connect to local Chroma data
            chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma-data")

            # EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
            # EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
            EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

            COLLECTION_NAME = 'cml-default'

            print("initialising Chroma DB connection...")

            print(f"Getting '{COLLECTION_NAME}' as object...")
            try:
                chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
                print("Success")
                collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
            except:
                print("Creating new collection...")
                collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
                print("Success")

            # Get latest statistics from index
            current_collection_stats = collection.count()
            print('Total number of embeddings in Chroma DB index is ' + str(current_collection_stats))
            
            context_chunk, source = get_nearest_chunk_from_chroma_vectordb(collection, message)
            response = get_llama2_response_with_context(message, context_chunk, temperature, token_count)
            response = f"{response}\n\n For additional info see source information below"
            bot_message = ""
            for i in range(len(response)):
                time.sleep(0.02)
                bot_message = response[:i+1]
                if i == len(response) - 1:
                    yield history + [(message, bot_message)], context_chunk
                else:
                    yield history + [(message, bot_message)], None
    # Handle other models if necessary

    
    elif model == "AWS Bedrock Claude v2.1":
        if vector_db == "None":
            # No context call Bedrock
            response = get_bedrock_response_with_context(message, "", temperature, token_count)
        
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                bot_message = response[:i+1]
                yield history + [(message, bot_message)], None  # Yield None for context_chunk
                
        elif vector_db == "Pinecone":
            
            PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
            PINECONE_INDEX = os.getenv('PINECONE_INDEX')

            print("initialising Pinecone connection...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            print("Pinecone initialised")

            print(f"Getting '{PINECONE_INDEX}' as object...")
            index = pc.Index(PINECONE_INDEX)
            print("Success")

            # Get latest statistics from index
            current_collection_stats = index.describe_index_stats()
            print('Total number of embeddings in Pinecone index is {}.'.format(current_collection_stats.get('total_vector_count')))
            
            # Vector search the index
            context_chunk, source, score = get_nearest_chunk_from_pinecone_vectordb(index, message)
            
            # Call Bedrock model
            response = get_bedrock_response_with_context(message, context_chunk, temperature, token_count)
            
            response = f"{response}\n\n For additional info see: {url_from_source(source)}"
            
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                bot_message = response[:i+1]
                if i == len(response) - 1:
                    print('string is',context_chunk[:4])
                    yield history + [(message, bot_message)], context_chunk
                else:
                    yield history + [(message, bot_message)], None
                
        elif vector_db == "Chroma":
            
            chroma_client = chromadb.PersistentClient(path="/home/cdsw/chroma-data")

            # EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"
            # EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
            EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

            COLLECTION_NAME = 'cml-default'

            print("initialising Chroma DB connection...")

            print(f"Getting '{COLLECTION_NAME}' as object...")
            try:
                chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
                print("Success")
                collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
            except:
                print("Creating new collection...")
                collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
                print("Success")

            # Get latest statistics from index
            current_collection_stats = collection.count()
            print('Total number of embeddings in Chroma DB index is ' + str(current_collection_stats))
            
            # Vector search in Chroma
            context_chunk, source = get_nearest_chunk_from_chroma_vectordb(collection, message)
            
            # Call CML hosted model
            response = get_bedrock_response_with_context(message, context_chunk, temperature, token_count)
            
            # Add reference to specific document in the response
            response = f"{response}\n\n For additional info see: {url_from_source(source)}"
            
            # Stream output to UI
            for i in range(len(response)):
                time.sleep(0.02)
                bot_message = response[:i+1]
                if i == len(response) - 1:
                    print('string is',context_chunk[:4])
                    yield history + [(message, bot_message)], context_chunk
                else:
                    yield history + [(message, bot_message)], None

def url_from_source(source):
    url = source.replace('/home/cdsw/data/https:/', 'https://').replace('.txt', '.html')
    return f"[Reference 1]({url})"
    

# Get embeddings for a user question and query Pinecone vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_pinecone_vectordb(index, question):
    # Generate embedding for user question with embedding model
    retriever = SentenceTransformer(EMBEDDING_MODEL_REPO)
    xq = retriever.encode([question]).tolist()
    xc = index.query(vector=xq, top_k=5,include_metadata=True)
    
    matching_files = []
    scores = []
    for match in xc['matches']:
        # extract the 'file_path' within 'metadata'
        file_path = match['metadata']['file_path']
        # extract the individual scores for each vector
        score = match['score']
        scores.append(score)
        matching_files.append(file_path)

    # Return text of the nearest knowledge base chunk 
    # Note that this ONLY uses the first matching document for semantic search. matching_files holds the top results so you can increase this if desired.
    response = load_context_chunk_from_data(matching_files[0])
    sources = matching_files[0]
    score = scores[0]
    
    print(f"Response of context chunk {response}")
    return response, sources, score
    #return "Cloudera is an Open Data Lakhouse company", "http://cloudera.com", 89 

# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()


# Get embeddings for a user question and query Chroma vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_chroma_vectordb(collection, question):
    ## Query Chroma vector DB 
    ## This query returns the two most similar results from a semantic search
    response = collection.query(
                    query_texts=[question],
                    # where={"metadata_field": "is_equal_to_this"}, # optional filter
                    # where_document={"$contains":"search_string"}  # optional filter
    )
    #print(results)
    
    return response['documents'][0][0], response['ids'][0][0]

def get_bedrock_response_with_context(question, context, temperature, token_count):
    
    # Supply different instructions, depending on whether or not context is provided
    if context == "":
        instruction_text = """Human: You are a helpful, honest, and courteous assistant. If you don't know the answer, simply state I don't know the answer to that question. Please provide an honest response to the user question enclosed in <question></question> tags. Do not repeat the question in the output.
    
    <question>{{QUESTION}}</question>
                    Assistant:"""
    else:
        instruction_text = """Human: You are a helpful, honest, and courteous assistant. If you don't know the answer, simply state I don't know the answer to that question. Please read the text provided between the tags <text></text> and provide an honest response to the user question enclosed in <question></question> tags. Do not repeat the question in the output.
    <text>{{CONTEXT}}</text>
    
    <question>{{QUESTION}}</question>
                    Assistant:"""

    
    # Replace instruction placeholder to build a complete prompt
    full_prompt = instruction_text.replace("{{QUESTION}}", question).replace("{{CONTEXT}}", context)
    
    # Model expects a JSON object with a defined schema
    body = json.dumps({"prompt": full_prompt,
             "max_tokens_to_sample":int(token_count),
             "temperature":float(temperature),
             "top_k":250,
             "top_p":1.0,
             "stop_sequences":[]
              })

    # Provide a model ID and call the model with the JSON payload
    modelId = 'anthropic.claude-v2:1'
    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept='application/json', contentType='application/json')
    response_body = json.loads(response.get('body').read())
    print("Model results successfully retreived")
    
    result = response_body.get('completion')
    #print(response_body)
    
    return result

    
# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llama2_response_with_context(question, context, temperature, token_count):
    
    llama_sys = f"<s>[INST]You are a helpful, respectful and honest assistant. If you are unsure about an answer, truthfully say 'I don't know'."
    
    if context == "":
        # Following LLama's spec for prompt engineering
        llama_inst = f"Please answer the user question.[/INST]</s>"
        question_and_context = f"{llama_sys} {llama_inst} [INST] {question} [/INST]"
    else:
        # Add context to the question
        #llama_inst = f"Anser the user's question based on the folloing information:\n {context}[/INST]</s>"
        llama_inst = f"Anser the user's question based on the folloing information: {context}[/INST]</s>"
        question_and_context = f"{llama_sys} {llama_inst} \n[INST] {question} [/INST]"
        
    try:
        # Build a request payload for CML hosted model
        data={ "request": {"prompt":question_and_context,"temperature":temperature,"max_new_tokens":token_count,"repetition_penalty":1.0} }
        
        r = requests.post(MODEL_ENDPOINT, data=json.dumps(data), headers={'Content-Type': 'application/json'})
        
        # Logging
        print(f"Request: {data}")
        print(f"Response: {r.json()}")
        
        no_inst_response = str(r.json()['response']['prediction']['response'])[len(question_and_context)-6:]
            
        return no_inst_response
        
    except Exception as e:
        print(e)
        return e


if __name__ == "__main__":
    main()
