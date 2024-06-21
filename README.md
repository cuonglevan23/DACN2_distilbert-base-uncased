    # Question Answering Demo

## Description

A simple web demo for Question Answering task: Given a question and a context, find the position (start and end index) of the answer within the context.

- Model: DistilBERT fine-tuned on SQuADv2 ([link](https://huggingface.co/thangduong0509/distilbert-finetuned-squadv2)).
- UI: Streamlit.

## How to use

### Option 1: Using conda environment

1. Create new conda environment and install required dependencies:

```
$ conda create -n <env_name> -y python=3.11
$ conda activate <env_name>
$ pip3 install -r requirements.txt
```

2. Host streamlit app

```
$ streamlit run app.py
```

### Option 2: Using docker

1. Build docker image

```
$ docker build -t <tag_name> -f docker/Dockerfile .
```

2. Run a docker contaier

```
$ docker run -p 8501:8501 -it <tag_name>
```
