import os
import json
import tiktoken
import openai
from dotenv import load_dotenv


# I stored the openai API key in a .env file because it is sensitive information
# Using the python-dotenv library we can access variables stored in the .env file
# load_dotenv will load in the variables from the .env file so we can then access them
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


file_path = './transcripts.json'

# Open the transcripts.json file in read mode
# Parses its contents into a dictionary called data
with open(file_path, 'r') as f:
    data = json.load(f) 


# Here we initialize tiktoken so we can tokenize and encode
# I used "cl100k_base" because I am using the gpt-3.5-turbo model
encoding = tiktoken.get_encoding("cl100k_base")

def process_chunk(chunk):
    # print(f"{len(chunk)} is the number of tokens in my text")
    # tokens_string = [encoding.decode_single_token_bytes(token) for token in chunk]
    # print(tokens_string)


    # This is the prompt I send along to openai with each chunk of conversation
    # It is constructed to retrieve any specific demographic information based on responses by the Pateint
    # It will also summarize health information based on the pateints response
    # The \n which we also see in the transcripts.json is a newline character, which in this case will ensure the chunk contents appears on a new line below the prompt
    prompt = "Extract demographic information about the patient, including their age, location, health, and any relevant personal details, from the following conversation between a doctor and a patient:\n" + encoding.decode(chunk)
    
    # Call the OpenAI API
    conversation = [{"role": "user", "content": prompt}]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # This is the path I took to get to our message in the response dictionary 
    assistant_response = response.choices[0].message['content']
    print(assistant_response)



# Define the maximum number of tokens per chunk
max_tokens = 100

# Iterate over each conversation transcript
for transcript in data['transcripts']:
    # Tokenize the transcript
    tokens = encoding.encode(transcript)
    
    # Initialize variables for chunking
    current_chunk = []
    current_length = 0
    current_chunk_number = 1

    for token in tokens:
        # Check if adding the token exceeds the max tokens per chunk
        if current_length < max_tokens:
            # Add the token to the current chunk
            current_chunk.append(token)
            current_length += 1
            
            # This is where I chose how to chunk the transcripts
            # I decided to chunk the transcripts after one full statement/question by the Doctor, and one full statement/question by Patient
            # After decoding the encoded chunk, I searched for b'?\n' and b'.\n' or b'.\n' twice, or b'?\n' twice as once this condition is met, one statement by each are found in the chunk and a new chunk should be started
            chunk_text = [encoding.decode_single_token_bytes(t) for t in current_chunk]
            if (b'?\n' in chunk_text and b'.\n' in chunk_text) or chunk_text.count(b'.\n') == 2 or chunk_text.count(b'?\n') == 2:

                begin_chunk_separator = f"[%--- Begin Chunk [Chunk {current_chunk_number}] ---%]"
                end_chunk_separator = f"[%--- End Chunk [Chunk {current_chunk_number}] ---%]"

                # Process the current chunk
                print(begin_chunk_separator)
                process_chunk(current_chunk)
                print(end_chunk_separator)
                
                # Start a new chunk
                current_chunk_number += 1
                current_chunk = []
                current_length = 0
                
        else:
            
            print(begin_chunk_separator)
            process_chunk(current_chunk)
            print(end_chunk_separator)

            current_chunk = [token]
            current_length = 1
            current_chunk_number += 1

    
    # Process the last chunk in the transcript
    
    print(begin_chunk_separator)
    process_chunk(current_chunk)
    print(end_chunk_separator)
    


