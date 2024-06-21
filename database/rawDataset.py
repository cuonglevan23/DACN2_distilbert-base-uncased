from datasets import load_dataset

def GetRawDataset():
    DATASET_NAME = 'squad_v2'
    raw_datasets = load_dataset(DATASET_NAME, split='train')
    raw_datasets = raw_datasets
    raw_datasets = raw_datasets.filter(
        lambda x: len(x['answers']['text']) > 0
    )

    return raw_datasets
