import jsonlines
from pprint import pprint
import pandas as pd

with jsonlines.open("/Users/kintch/PycharmProjects/gpt_comparativ_judgement/result_data.jsonl") as f:
    lines = [x for x in f.iter()]

parallel_jurisdiction_list=[]
for k in range(len(lines)):
    check_if_noerror = type(lines[k][1]) == type(dict())
    if check_if_noerror == True:
        input = lines[k][0]['messages'][1]['content']
        input_words_list = [x.strip() for x in input.split(',')]
        result = lines[k][1]['choices'][0]['message']['content']
        parallel_jurisdiction_list.append(input_words_list+[result])

pd.DataFrame(parallel_jurisdiction_list,
             columns=['word1', 'word2', 'easier_word']).to_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/test_3.xlsx')


