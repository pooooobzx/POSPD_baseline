import nltk
from collections import defaultdict
import re
import pickle as pkl
import time
nltk.download('averaged_perceptron_tagger')
def read_txt(file):
    f = open(file,"r",encoding="utf-8")
    text = []
    data = f.read().splitlines()
    for line in data:
        line = line.split()
        text.append(line)
    return text
limit_data = read_txt("pos_limit100.txt")
l_keys = []
l_ids = []
for line in limit_data:
    l_keys.append(line[0])
    l_ids.append(line[1])
limit_dict = dict(zip(l_keys,l_ids))


def save_txt(text,file):
    f = open(file,"w",encoding="utf-8")
    for line in text:
        f.write(" ".join(line)+"\n")
    f.close()
    print("write file to:"+ file)

def process_bpe(text):
    #recover sentences from bpe sentences
    clean =[]
    for line in text:
        a = " ".join(line)
        clean.append(" ".join(line).replace(" ##","").split())
    return clean

def pos_subid(pos_tag,ids):
    a = int(limit_dict[pos_tag])
    if a==0:
        return pos_tag
    elif ids < a:
        return pos_tag+str(ids)
    else:
        return pos_tag+str(a)

def getpos_bpe(data1,data2):
    """
    data1 = list of bpe sentences
    data2 = list of raw sentences
    """
    pos_test = []
    time1 = time.time()
    for i in range(len(data1)):
        if i %100000 ==1:
            print(i)
            print(time.time()-time1)
            print("#"*66)
            time1=time.time()
        pos = []
        line = data1[i]
        pos_line = nltk.pos_tag(data2[i])
        p_line = []
        for w,p in pos_line:
            if w == p or w in [":","?","-","...",";","--","!"] or p in ["(",")","``"]:
                p_line.append("PCT")  
            else:
                #combining "(",")"and"``" are named as PCT,
                #"''"and"$" are named as SYM$, WP$ is combined to WP
                if p in ["$","''","SYM"]:
                    p_line.append("SYM$")
                elif p=="WP$":
                    p_line.append("WP")
                else:
                    p_line.append(p)
        j=0
        n = 0

        while j < len(line):
            if j==len(line)-1:
                pos.append(p_line[n])
                break
            else:
                if "##" not in line[j+1]:
                    pos.append(p_line[n])
                    j+=1
                else:
                    k=1
                    while k:
                        if j+k==len(line): 
                            pos.append(pos_subid(p_line[n],k))
                            break
                        pos.append(pos_subid(p_line[n],k))
                        if "##" not in line[j+k]:
    #                         pos.append(pos_subid(p_line[n],k+1))
                            break
                        k+=1      
                    j = j+k
                n+=1

        assert len(pos)==len(line)
        pos_test.append(pos) 
    return pos_test
train_data1 = read_txt("train.tgt")#bpe
train_data2 = process_bpe(train_data1) #without bpe
train_pos = getpos_bpe(train_data1,train_data2)
save_txt(train_pos,"pos_train.tgt")