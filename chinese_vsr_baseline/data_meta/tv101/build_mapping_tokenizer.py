
import csv
from collections import Counter
import re


special_tokens = {
    '<ig>': -1,
    '<sos>': 0,
    '<eos>': 1,
    '<blank>': 2
}

with open('../../../data/CAS-VSR-S101/labels/CAS-VSR-S101_trans.csv', 'r', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader) 
    words = []
    for row in csv_reader:
        print(row)
        if len(row)==4:
            _, _, _, trans = row
        cleaned_text = re.sub(r'[^\w\u4e00-\u9fff]', '', trans)
        for word in cleaned_text:
            words.append(word)


word_freq = Counter(words)


for special_token in special_tokens.keys():
    word_freq.pop(special_token, None)


sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
vocab = {word: index for word, index in special_tokens.items()}


for word, _ in sorted_words:
    vocab[word] = len(vocab)-1

with open('mapping_tokenizer.txt', 'w', encoding='utf-8') as vocab_file:
    for word, index in sorted(vocab.items(), key=lambda x: x[1]):
        vocab_file.write(f"{word},{index}\n")
