from bpemb import BPEmb
def conform(sen):
    for i, each_word in enumerate(sen):
        if each_word.isalpha():
            sen[i] = " ##" + each_word  
        elif each_word[0] == '▁':
            sen[i] = each_word[1:]           
def main():
    filename = "train.summary"
    output_name = "gabbage.tgt"
    f_out  = open(output_name, "w")
    bpemb_en = BPEmb(lang="en", vs = 200000)
    f = open(filename, 'r')
    lines = f.readlines()
    for each_line in lines:
        k = bpemb_en.encode(each_line)
        
        conform(k)
        if (len(k) != len(bpemb_en.encode_ids(each_line))):
            print("False")
        to_write = " ".join(k)
        f_out.write(to_write)
    f_out.close()
    f.close()
        
        

main()