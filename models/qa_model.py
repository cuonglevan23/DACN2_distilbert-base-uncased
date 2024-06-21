from transformers import pipeline

def get_model(model_id):
    PIPELINE_NAME = 'question-answering'
    pipe = pipeline(PIPELINE_NAME, model=model_id, device='cpu')

    return pipe