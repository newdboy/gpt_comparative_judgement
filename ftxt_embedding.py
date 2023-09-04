from gensim.models import FastText
import numpy as np
from soynlp.hangle import compose, decompose, character_is_korean
import re


model_fname = '/Users/kintch/Downloads/jamoed_wiki_n_namu_based_model'
model = FastText.load(model_fname)

def jamo_sentence(sent):
    def transform(char):
        if char == ' ':
            return char
        cjj = decompose(char)
        if len(cjj) == 1:
            return cjj
        cjj_ = ''.join(c if c != ' ' else '-' for c in cjj)
        return cjj_

    sent_ = []
    for char in sent:
        if character_is_korean(char):
            sent_.append(transform(char))
        else:
            sent_.append(char)
    sent_ = doublespace_pattern.sub(' ', ''.join(sent_))
    return sent_

def jamo_to_word(jamo):
    jamo_list, idx = [], 0
    while idx < len(jamo):
        if not character_is_korean(jamo[idx]):
            jamo_list.append(jamo[idx])
            idx += 1
        else:
            jamo_list.append(jamo[idx:idx + 3])
            idx += 3
    word = ""
    for jamo_char in jamo_list:
        if len(jamo_char) == 1:
            word += jamo_char
        elif jamo_char[2] == "-":
            word += compose(jamo_char[0], jamo_char[1], " ")
        else:
            word += compose(jamo_char[0], jamo_char[1], jamo_char[2])
    return word

def transform(list):
    return [(jamo_to_word(w), r) for (w, r) in list]

doublespace_pattern = re.compile('\s+')

def similar_words(word, k):
    jamoed_word = jamo_sentence(word)
    result = transform(model.wv.most_similar(jamoed_word, topn=k))
    return result

# similar_words('1960년 트리핀', 10)
# similar_words('당시 대규모 대미 무역', 10)
# jamo_sentence('1960년트리핀')

def cosine_similarity(word1, word2):
    cjj1 = jamo_sentence(word1)
    cjj2 = jamo_sentence(word2)
    cos_sim = model.wv.similarity(cjj1, cjj2)
    return cos_sim