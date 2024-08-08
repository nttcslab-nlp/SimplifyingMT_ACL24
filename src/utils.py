import math

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import spacy
from tqdm import tqdm

def tokenize_sentences_to_words(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        lowercase_sentence = sentence.lower()
        tokenized_sentences.append(word_tokenize(lowercase_sentence))
    return tokenized_sentences

def read_lines_from_file(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def df_to_src_ref(json_path):
    df = pd.read_json(json_path,lines=True)
    sources = df['source'].to_list()
    references = df['reference'].to_list()
    return sources,references

def create_aoa_dict():
    aoa_dict = {}
    path = "../data/aoa.txt"
    with open(path) as f:
        for line in f:
            line = line.split("\t")
            if line[10]== 'NA':
                aoa_dict[line[0]] = 0
            else:
                aoa_dict[line[0]] = float(line[10])
    return aoa_dict

def count_aoa(aoa_lists):
    aoa_count = {x:0 for x in range(25)}
    for aoa_values in aoa_lists:
        if aoa_values:
            floor_aoa = math.floor(max(aoa_values))
            aoa_count[floor_aoa] += 1
    return list(aoa_count.values())

def average_aoa(aoa_lists):
    total = 0
    for aoa_values in aoa_lists:
        if aoa_values:
            max_aoa = max(aoa_values)
            total += max_aoa
    return total/len(aoa_lists)

# Calculate AoA for each word in a sentence
# Calculate by considering '-' and proper nouns within a word
def get_aoa(sentences,AoA):
    nlp = spacy.load("en_core_web_sm")
    out_words,out_aoa = [],[]
    for sentence in tqdm(sentences,total = len(sentences)):
        tmp_words,tmp_aoas=[],[]
        sentence = sentence.replace(" - "," ")
        doc = nlp(sentence) 
        tmp,prev_is_propn = 0,0
        try:
            for token in doc:
                if token.pos_ == "PROPN":
                    if tmp == 0:
                        prev_is_propn=1
                    else:
                        tmp = 0
                else:
                    text, lemma = token.text, token.lemma_
                    if tmp == 0:
                        if text in AoA:
                            tmp_aoas.append(AoA[text])
                            tmp_words.append(text)
                        elif lemma in AoA:
                            tmp_aoas.append(AoA[lemma])
                            tmp_words.append(text)
                        elif text == "-":
                            if prev_is_propn == 1:
                                pass
                            else:
                                tmp += 1
                        else:
                            tmp_aoas.append(0)
                            tmp_words.append(text)
                    else:
                        if text in AoA:
                            aoa = AoA[text]
                            if aoa >= tmp_aoas[-1]:
                                tmp_aoas[-1] = aoa
                        elif lemma in AoA:
                            aoa = AoA[lemma]
                            if aoa >= tmp_aoas[-1]:
                                tmp_aoas[-1] = aoa
                        tmp_words[-1] = tmp_words[-1] + "-" + text
                        tmp = 0
                    prev_is_propn = 0
            out_words.append(tmp_words)
            out_aoa.append(tmp_aoas)
        except Exception as e:
            print(f"Error processing sentence: {sentence} with error: {e}")
    return out_words,out_aoa
