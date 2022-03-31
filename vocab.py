from tqdm import tqdm
from collections import Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import nltk
from nltk.data import load


class Vocabulary:
    def tokenizer(self, text):
        return text.lower().split(" ")

    def __init__(self):
        self.itos = {0 :"<PAD>", 1:"<UNK>"}
        self.stoi = {"<PAD>":0, "<UNK>":1}
        train_source_text = list(open("gigaword_8/train.article", "r"))
    # We also consider the test target as we do not want to replace any words in it by unk token.
        test_target_text = list(open("gigaword_8/test.article"))
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
        idx =2
        for key, value in list(d.most_common()):
            self.stoi[key]=idx 
            
            self.itos[idx]=key
            idx  += 1 
        print("vocab initizalied")
    def __len__(self):
        return len(self.itos)
    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return[self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]



class PosVocab:
    def __init__(self):
        tagdict = load('help/tagsets/upenn_tagset.pickle')
        all_strings = list(tagdict.keys())
        self.itos = {0:"<SOS>", 1:"<UNK>"}
        self.stoi = {"<SOS>":0, "<UNK>":1}
        idx = 2
        for each in all_strings:
            self.stoi[each]=idx 
            
            self.itos[idx]=each
            idx  += 1 
    def __len__(self):
        return len(self.itos)
    def tokenizer(self, text):
        return text.lower().split(" ")
    def nummericalize_text(self, text):
        #text : string 
        tokenized_text = self.tokenizer(text)
        tokenized_text = nltk.pos_tag(tokenized_text)
        return[self.stoi[p] if p in self.stoi else self.stoi["<UNK>"] for w, p  in tokenized_text]
    def nummericalize_Tag(self, tagging):
        #tagging: list of tags alone
        return [self.stoi[p] if p in self.stoi else self.stoi["<UNK>"] for p  in tagging]


        




