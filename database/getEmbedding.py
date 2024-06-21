from transformers import AutoTokenizer, AutoModel
import torch
from database import rawDataset

import numpy as np
import sqlite3

def CreateDB(embeddings_dataset, db_file='embeddings.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY,
                        question TEXT,
                        context TEXT,
                        question_embedding BLOB
                    )''')

    # Insert data from embeddings dataset
    for idx, embedding_entry in enumerate(embeddings_dataset):
        question_text = embedding_entry['question']
        context_text = embedding_entry['context']
        question_embedding = np.array(embedding_entry['question_embedding'])  # Ensure it's a numpy array
        cursor.execute('''INSERT INTO embeddings (question, context, question_embedding) VALUES (?, ?, ?)''', 
                       (question_text, context_text, question_embedding.tobytes()))

    # Commit changes and close connection
    conn.commit()
    conn.close()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

raw_datasets = rawDataset.GetRawDataset()


MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)

    return cls_pooling(model_output)



def BuildVectorDB():
    # Convert to numpy array (required for HF Datasets)
    EMBEDDING_COLUMN = 'question_embedding'
    embeddings_dataset = raw_datasets.map(
        lambda x: {EMBEDDING_COLUMN: get_embeddings(x['question']).detach().cpu().numpy()[0]}
    )

    embeddings_dataset.add_faiss_index(column=EMBEDDING_COLUMN)
    return embeddings_dataset

