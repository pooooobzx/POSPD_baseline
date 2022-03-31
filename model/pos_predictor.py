from tqdm import tqdm
from collections import Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class MyDataSet(Dataset):
    def __init__(self):
        train_source_text = list(open("gigaword_8/train.article", "r"))
    # We also consider the test target as we do not want to replace any words in it by unk token.
        test_target_text = list(open("gigaword_8/test.summary"))
        #        Combine two text file together
        total_text = train_source_text + test_target_text
        self.text = total_text

        d = dict()

# Loop through each line of the file
        for line in tqdm(total_text):
            # Remove the leading spaces and newline character
            line = line.strip()

            # Convert the characters in line to
            # lowercase to avoid case mismatch
            line = line.lower()

            # Split the line into words
            words = line.split(" ")

            # Iterate over each word in line
            for word in words:
                # Check if the word is already in dictionary
                if word in d:
                    # Increment count of word by 1
                    d[word] = d[word] + 1
                else:
                    # Add the word to dictionary with count 1
                    d[word] = 1
        d = Counter(d)

        

