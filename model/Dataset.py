from vocab import Vocabulary, PosVocab
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
class MyDataSet(Dataset):
    def __init__(self, k):
        train_source_text_file = open("gigaword_{}/train.article".format(k), "r")
        train_target_text_file = open("gigaword_{}/train.summary".format(k), "r")
        src = train_source_text_file.readlines()
        tgt = train_target_text_file.readlines()
        data_zip = zip(src, tgt)
        data_zip = list(data_zip)
        self.final_data = []
        for item in data_zip:
            if len(item[1].strip('\n').split(" ")) != k:
                continue
            self.final_data.append([item[0].strip("\n"), item[1].strip("\n")])
        self.vocab = Vocabulary()
        self.tag_vocab = PosVocab()
    def __len__(self):
        return len(self.final_data)
    def __getitem__(self, index):
        data = self.final_data[index]
        numerical_src = self.vocab.numericalize(data[0])
        tgt_input = nltk.pos_tag(data[1].lower().split()[:-1])
        tgt_input = [p for w, p in tgt_input]
        tgt_input = ["<SOS>"] + tgt_input
        numerical_tgt_input = self.tag_vocab.nummericalize_Tag(tgt_input)
        
        numerical_tgt_output1 = self.tag_vocab.nummericalize_text(data[1])

        numerical_tgt_output2 = self.vocab.numericalize(data[1])
        
        return torch.tensor(numerical_src), torch.tensor(numerical_tgt_input), torch.tensor(numerical_tgt_output1), torch.tensor(numerical_tgt_output2)
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        encode_inputs = [item[0] for item in batch]
        decode_inputs = [item[1].tolist() for item in batch]
        decode_outputs1 = [item[2].tolist() for item in batch]
        decode_outputs2 = [item[3].tolist() for item in batch]

        encode_inputs = pad_sequence(encode_inputs, batch_first=True, padding_value=self.pad_idx)

        return encode_inputs,torch.tensor(decode_inputs), torch.tensor(decode_outputs1), torch.tensor(decode_outputs2)

def get_loader(k, batch_size = 512, shuffle=True, pin_memory=True):
    dataSet = MyDataSet(k)
    pad_idx = dataSet.vocab.stoi["<PAD>"]
    assert pad_idx == 0
    loader= DataLoader(dataset = dataSet, batch_size = batch_size, shuffle = shuffle, pin_memory = pin_memory, collate_fn = MyCollate(pad_idx)) #drop_last = True )

    return loader, dataSet

