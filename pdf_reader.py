import os
import re
import shutil
from pdfminer.high_level import extract_text
import uuid


#####################
## updates 9/16:
## remove the workflow that removes prior data chunk files prior to creating new ones
## New flow will continue to add .txt chuncks incrementally
#####################



# Configuration and path settings
staged_folder = 'staged_files'  # Folder where files are staged for processing
completed_folder = 'staged_files_completed'  # Folder to move processed files
output_folder_txt = 'output_text_files'  # Folder to store extracted text files (not used in code)
output_folder_data = 'data'  # Folder to store chunked files

### staged for removal
#old_data_folder = 'old_data'  # Folder to store old data for backup

def extract_text_from_pdfs(stage_folder, completed_folder):
    for pdf_file in os.listdir(stage_folder):
        try:
            if pdf_file.endswith(".pdf"):
                print(f"Processing file: {pdf_file}")  # Debug: File being processed
                
                output_file_name = os.path.splitext(pdf_file)[0] + ".txt"
                output_path = os.path.join(staged_folder, output_file_name)
                pdf_path = os.path.join(stage_folder, pdf_file)
                
                # Extracting text
                text = extract_text(pdf_path)
                print(f"Text extracted from {pdf_file}, length: {len(text)}")  # Debug: Length of extracted text
                
                # Adding space at the end of each line
                text = "\n".join([line.strip() + " " for line in text.splitlines()])
                
                # Writing extracted text to a file
                with open(output_path, "w", encoding="utf-8") as out:
                    out.write(text)
                print(f"Written text to {output_file_name}")  # Debug: Text written to file
                
                # Moving the processed PDF to the completed folder
                shutil.move(pdf_path, os.path.join(completed_folder, pdf_file))
                print(f"Moved {pdf_file} to {completed_folder}")  # Debug: File moved to completed folder
            
        except Exception as e:
            print(f"Error processing file {pdf_file}: {e}")  # Debug: Error encountered
            continue  # Continue with the next file if an error occurs


def process_text(text):
    # Clean up text by removing non-ASCII characters and unwanted symbols
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = re.sub(r'[^A-Za-z0-9 .,?!]+', '', text)  # Remove unwanted characters
    return text

def chunk_file_and_save(filename):
    # Split the text into chunks of 500 words and save each chunk as a new file
    with open(filename, 'r') as f:
        content = f.read()

    content = process_text(content)  # Clean up the text
    word_list = content.split()  # Split text into words
    start_idx = 0

    # Loop through the text and chunk it
    while start_idx < len(word_list):
        end_idx = start_idx + 500

        # Ensure that we are not going beyond the length of the word list
        if end_idx >= len(word_list):
            end_idx = len(word_list) - 1

        # Try to adjust to the nearest punctuation mark, but only within the chunk range
        for i in range(end_idx, start_idx, -1):
            if word_list[i][-1] in ['.', '!', '?']:
                end_idx = i
                break

        # Create the chunk
        chunk = ' '.join(word_list[start_idx:end_idx + 1])  # Include end_idx in chunk

        # Ensure the chunk is not empty
        if chunk.strip():  # Check if chunk contains non-whitespace text
            unique_filename = f"doc_{uuid.uuid4()}.txt"
            with open(os.path.join(output_folder_data, unique_filename), 'w') as output_file:
                output_file.write(chunk)
                print(f'Writing: {unique_filename}')
        else:
            print(f"Skipping empty chunk at index {start_idx}")

        # Move the start index for the next chunk
        start_idx = end_idx + 1

        # If fewer than 500 words left, we're at the end
        if end_idx >= len(word_list) - 1:
            print("Reached the end of the word list.")
            break

### staged for removal

# def delete_old_data():
#     # Delete old data files and directories in the old_data_folder
#     for filename in os.listdir(old_data_folder):
#         file_path = os.path.join(old_data_folder, filename)

#         if os.path.isfile(file_path):
#             os.remove(file_path)  # Remove the file
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)  # Remove the directory and all its contents

if __name__ == '__main__':
    # Main processing logic
    # Ensure output folder and completed folders exist
    for dir_name in [output_folder_data, completed_folder]:
        if not os.path.exists(dir_name):
            print("Trying to create:", dir_name)
            os.makedirs(dir_name)
            
    extract_text_from_pdfs(staged_folder, completed_folder)  # Extract text from PDFs

    ## staged for removal
    
    # Move existing files in the output folder to old data folder
    # for filename in os.listdir(output_folder_data):
    #     dst_path = os.path.join(old_data_folder, filename)
    #     if os.path.exists(dst_path):
    #         os.remove(dst_path)
    #     shutil.move(os.path.join(output_folder_data, filename), dst_path)

    # Check if the staged folder exists
    if not os.path.exists(staged_folder):
        raise FileNotFoundError("The 'staged_files' directory does not exist.")

    # Get a list of .txt files in the staged folder
    txt_files = [f for f in os.listdir(staged_folder) if f.endswith('.txt')]

    # If no .txt files found, raise an error
    if not txt_files:
        raise FileNotFoundError("No .txt files found in the 'staged_files' directory.")

    # Process each .txt file by chunking it into smaller files
    for file in txt_files:
        print(file)
        chunk_file_and_save(os.path.join(staged_folder, file))
        os.remove(os.path.join(staged_folder, file))  # Remove the processed .txt file