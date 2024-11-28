'''
    structuring finetuning dataset

    For now, try with the following priority list:
    1. Structure finetuning dataset with images from design team
    2. If there is significant performance difference, create synthetic data using images design team gave us as reference.

    TODO: create synthetic data


'''

# for importing the finetuning dataset to Hugging Face

from datasets import load_dataset



dataset = load_dataset("imagefolder", data_files="vapor.zip", split="train")
dataset.push_to_hub("vaporwave-dataset")




