from curses import echo
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from Dataset import MyDataSet, get_loader
from untitled20 import Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader, dataSet = get_loader(8)
src_vocab_size = len(dataSet.vocab)
tgt_vocab_size = len(dataSet.tag_vocab)
d_model = 512
d_ff = 2048
d_k = d_v = 64
Encoder_layer = 12
Decoder_layer = 1
n_heads = 8


model = Transformer(src_vocab_size, tgt_vocab_size, d_model, Encoder_layer, Decoder_layer, d_k, d_v, n_heads,d_ff).to(device)
criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay = 1e-5)
for epoch in range(50):
    for enc_inputs, dec_inputs, dec_outputs, dec_outputs2 in loader:
        print(enc_inputs.shape)
        
        enc_inputs, dec_inputs, dec_outputs, dec_outputs2 = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device), dec_outputs2.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns, outputs2= model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1)) + criterion2(outputs2, dec_outputs2.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
checkpont={'epoch': 101, "state_dict" : model.state_dict(), "optimizer":optimizer.state_dict()}
torch.save(checkpont, "best_1_layer_optimized_loss.pt")

def greedy_decoder(model, enc_input, start_symbol):
    """
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    next_symbol = start_symbol
    while not dec_input.shape[-1] == 9 :         
        dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).to(device)],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        print(next_word)            
    return dec_input
# Test
loader, dataSet = get_loader(8)
enc_inputs, _, _, _ = next(iter(loader))
enc_inputs = enc_inputs.to(device)


for i in range(len(enc_inputs)):
    greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=0)
    print(enc_inputs[i], '->', [dataSet.tag_vocab.itos[n.item()] for n in greedy_dec_input.squeeze()])