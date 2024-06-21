import os
import base64
import streamlit as st
from annotated_text import annotated_text

from components.streamlit_footer import footer
from config.model_config import QA_Config
from models.qa_model import get_model
from database import getEmbedding
import sqlite3
import numpy as np
import random


EXAMPLE_QUESTIONS = [
    "What doctor was with Chopin when he wrote out his will?",
    "Where was Chopin invited to in late summer?",
    "What city did Chopin perform at on September 27?",
    "What did Chopin write while staying with Doctor Adam Łyszczyński?",
    "When did Chopin last appear in public?",
    "Who were the beneficiaries of his last public concert?",
    "What was the diagnosis of Chopin's health condition at this time?",
    "Where was Chopin's last public performance?",
    "Who did Chopin play for while she sang?",
    "In 1849 where did Chopin live?",
    "Who was anonymously paying for Chopin's apartment?",
    "When did Chopin return to Paris?",
    "Chopin accompanied which singer for friends?",
    "Where did his friends found Chopin an apartment in 1849?",
    "Who paid for Chopin's apartment in Chaillot?",
    "When did Jenny Lind visit Chopin?",
    "When did his sister come to stay with Chopin?"
]

EXAMPLE_QUESTION = ""
def replace_input_text():
    EXAMPLE_QUESTION = random.choice(EXAMPLE_QUESTIONS)
    st.session_state.question = EXAMPLE_QUESTION

def get_answer(question, context):
    qa_model = get_model(QA_Config.model_id)
    result_dict = qa_model(question=question, context=context)
    
    answer_text = result_dict['answer'] if 'answer' in result_dict else "No answer found"
    answer_start = result_dict['start'] if 'start' in result_dict else -1
    answer_end = result_dict['end'] if 'end' in result_dict else -1
    
    start_ans_idx = answer_start
    end_ans_idx = answer_end
    
    return [
        context[:start_ans_idx],
        (context[start_ans_idx:end_ans_idx], '', '#afa'),
        context[end_ans_idx:]
    ]




# def get_nearest_examples(query_embedding, k=1):
#     # Connect to the SQLite database
#     conn = sqlite3.connect('embeddings.db')
#     cursor = conn.cursor()

#     # Perform similarity search
#     cursor.execute('''SELECT id, question, question_embedding FROM embeddings''')
#     all_embeddings = cursor.fetchall()

#     # Calculate similarity scores (Euclidean distance)
#     similarities = [(idx, np.linalg.norm(np.frombuffer(embedding) - query_embedding)) for idx, _, embedding in all_embeddings]

#     # Sort by similarity and return the top k nearest examples
#     nearest_examples = sorted(similarities, key=lambda x: x[1])[:k]

#     # Close connection
#     conn.close()

#     return nearest_examples

def get_nearest_contexts(query_embedding, db_file='embeddings.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Fetch data from the database
    cursor.execute('''SELECT context, question_embedding FROM embeddings''')
    all_embeddings = cursor.fetchall()

    # Calculate similarity scores (Euclidean distance)
    similarities = [(context, np.linalg.norm(np.frombuffer(embedding) - query_embedding))
                    for context, embedding in all_embeddings]

    # Sort by similarity and get the nearest context
    nearest_context = min(similarities, key=lambda x: x[1])[0]  # Get the context with the minimum distance

    # Close connection
    conn.close()

    return nearest_context


def main():
    st.set_page_config(page_title="Question Answering Demo - AI VIETNAM",
                       page_icon='static/aivn_favicon.png',
                       layout="wide")
    
    db_file = 'embeddings.db'
    if not os.path.exists(db_file):
        # If the database file doesn't exist, create it and populate with embeddings dataset
        embeddings_dataset = getEmbedding.BuildVectorDB()
        getEmbedding.CreateDB(embeddings_dataset, db_file)
    else:
        print("Database file already exists.")

    col1, col2 = st.columns([0.8, 0.2], gap='large')
    
    with col1:
        st.title(':thought_balloon: :blue[Question Answering] Demo')
        
    # with col2:
        # logo_img = open("static/aivn_logo.png", "rb").read()
        # logo_base64 = base64.b64encode(logo_img).decode()
        # st.markdown(
        #     f"""
        #     <a href="https://aivietnam.edu.vn/">
        #         <img src="data:image/png;base64,{logo_base64}" width="full">
        #     </a>
        #     """,
        #     unsafe_allow_html=True,
        # )
        
    # input_context = EXAMPLE_CONTEXT

    with st.form("my_form"):
        input_question = st.text_input('__Question__',
                                        key='question',
                                        max_chars=100,
                                        placeholder='Input some text...')
        
        # context_input = st.text_area('__Context__',
        #                              height=100,
        #                              max_chars=1000,
        #                              key='context',
        #                              value=input_context,
        #                              disabled=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            submission = st.form_submit_button('Submit')
        with col2:
            example_button = st.form_submit_button('Run example', on_click=replace_input_text)
            
        if example_button:
            st.divider()
            if input_question == '':
                st.write('__Error:__ Input question cannot be empty!')
            else:
                input_quest_embedding = getEmbedding.get_embeddings([input_question]).cpu().detach().numpy()[0]

                nearest_contexts = get_nearest_contexts(input_quest_embedding)
                
                if nearest_contexts:
                    nearest_context = nearest_contexts  # Assuming the second element is the nearest context
                    st.text_area("The context for your question: ", nearest_context)
                    result_lst = get_answer(input_question, nearest_context)
                    st.write(f'__Answer__: {result_lst[1][0]}')
                    annotated_text(result_lst)
                else:
                    st.write('No similar context found in the database.')
                    

        if submission:
            st.divider()
            if input_question == '':
                st.write('__Error:__ Input question cannot be empty!')
            else:
                input_quest_embedding = getEmbedding.get_embeddings([input_question]).cpu().detach().numpy()[0]

                nearest_contexts = get_nearest_contexts(input_quest_embedding)
                
                if nearest_contexts:
                    nearest_context = nearest_contexts  # Assuming the second element is the nearest context
                    st.text_area("The context for your question: ", nearest_context)
                    result_lst = get_answer(input_question, nearest_context)
                    st.write(f'__Answer__: {result_lst[1][0]}')
                    annotated_text(result_lst)
                else:
                    st.write('No similar context found in the database.')


    footer()

if __name__ == '__main__':
    main()


# def main():
#     st.set_page_config(page_title="Question Answering Demo - AI VIETNAM",
#                        page_icon='static/aivn_favicon.png',
#                        layout="wide")
    
    
#     db_file = 'embeddings.db'
#     if not os.path.exists(db_file):
#         # If the database file doesn't exist, create it and populate with embeddings dataset
#         embeddings_dataset = getEmbedding.BuildVectorDB()
#         getEmbedding.CreateDB(embeddings_dataset, db_file)
#     else:
#         print("Database file already exists.")

#     col1, col2 = st.columns([0.8, 0.2], gap='large')
    
#     with col1:
#         st.title(':thought_balloon: :blue[Question Answering] Demo')
        
#     with col2:
#         logo_img = open("static/aivn_logo.png", "rb").read()
#         logo_base64 = base64.b64encode(logo_img).decode()
#         st.markdown(
#             f"""
#             <a href="https://aivietnam.edu.vn/">
#                 <img src="data:image/png;base64,{logo_base64}" width="full">
#             </a>
#             """,
#             unsafe_allow_html=True,
#         )
        
        
#     input_context = "Will match with your question"
#     with st.form("my_form"):
#         input_question = st.text_input('__Question__',
#                                         key='question',
#                                         max_chars=100,
#                                         placeholder='Input some text...')
        
#         context_input = st.text_area('__Context__',
#                                      height=100,
#                                      max_chars=1000,
#                                      key='context',
#                                      value=input_context,
#                                      disabled=True)
        
#         col1, col2 = st.columns([1, 2])
#         with col1:
#             submission = st.form_submit_button('Submit')
#         with col2:
#             example_button = st.form_submit_button('Run example', 
#                                                    on_click=replace_input_text)
            
#         if example_button:
#             st.divider()
#             result_lst = get_answer(EXAMPLE_QUESTION, EXAMPLE_CONTEXT)
#             st.write(f'__Answer__: {result_lst[1][0]}')
#             annotated_text(result_lst)

#         if submission:
#             st.divider()
#             if input_question == '':
#                 st.write('__Error:__ Either input question or context cannot be empty!')
#             else:
#                 input_quest_embedding = getEmbedding.get_embeddings([input_question]).cpu().detach().numpy()

#                 # _, samples = embeddings_dataset.get_nearest_examples(
#                 # "question_embedding", input_quest_embedding, k=1
#                 # )
                
#                 samples = get_nearest_examples(input_quest_embedding, k=1)

#                 input_context = samples["context"]
                
#                 st.text_area.write("The context for your question: ", input_context)
#                 result_lst = get_answer(input_question, input_context)
#                 st.write(f'__Answer__: {result_lst[1][0]}')
#                 annotated_text(result_lst)


#     footer()


# if __name__ == '__main__':
#     main()