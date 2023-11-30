import json
from collections import OrderedDict
import concurrent.futures
from tqdm import tqdm
import numpy as np
from itertools import product, combinations
import word_variable

def extract_random_words_from_list(word_list, n, random_seed=0):
    if n > len(word_list):
        print("n is bigger than length of word_list")
        return None
    else:
        print(f"n(:{n}) is not bigger than length of word_list(:{len(word_list)})")
        np.random.seed(random_seed)
        result = list()
        for k in tqdm(np.random.choice(range(0, len(word_list)), n, replace = False)):
            result.append(word_list[k])
        return result

def make_comparing_word_list(word_list):
    result = list()
    for word1, word2 in tqdm(list(product(*[word_list, word_list]))):
        if word1 != word2:
            df = word_variable.df_ku
            aoa_word1 = df['Rating.Mean'][df.Word == word1].tolist()[0]
            aoa_word2 = df['Rating.Mean'][df.Word == word2].tolist()[0]
            # print(aoa_word1, aoa_word2)
            if aoa_word1 < aoa_word2:
                assist_output = word1
            else:
                assist_output = word2
            # print(word1, ":", aoa_word1, "&", word2, ":", aoa_word2, "->", assist_output)
            data_line = OrderedDict()
            # data_line["model"] = "gpt-3.5-turbo"
            data_line["messages"] = [{"role": "system",
                                      "content": "You're a language teaching professional with a keen sense of word level. You are to choose between two words: {word A, word B}. Pick the word you believe is learned first in childhood. When answering: 1.Choose the earlier-learned word. 2. Only state the word, nothing more."
                                      },
                                     {"role": "user",
                                      "content": str(word1) + ", " + str(word2)
                                      },
                                     {"role": "assistant",
                                      "content": str(assist_output)
                                      },
                                     ]
            result.append(data_line)

    return result

def make_comparing_word_list_base_mdl(word_list):
    result = list()
    for word1, word2 in tqdm(list(itertools.product(*[word_list, word_list]))):
        if word1 != word2:
            df = word_variable.df_ku
            aoa_word1 = df['Rating.Mean'][df.Word == word1].tolist()[0]
            aoa_word2 = df['Rating.Mean'][df.Word == word2].tolist()[0]
            # print(aoa_word1, aoa_word2)
            if aoa_word1 < aoa_word2:
                assist_output = word1
            else:
                assist_output = word2
            # print(word1, ":", aoa_word1, "&", word2, ":", aoa_word2, "->", assist_output)
            # data_line["model"] = "gpt-3.5-turbo"
            result.append({
                "prompt": f"The easier word between '{word1}' and '{word2}' is",
                "completion": f" '{assist_output}'."
            })

    return result

def make_word_pair_list(word_list, lang):
    """
    :param word_list: (ko) [easier_word_list, more_difficult_word_list]
                      (en) word_list
    :param lang:
    :return:
    """

    if lang == "ko":
        """if word_list =[1, 2], [3,4], return [(1,3), (1,4), (2,3), (2,4)]"""
        easier_word_list, diff_word_list = word_list[0], word_list[1]
        return list(product(*[easier_word_list, diff_word_list]))

    elif lang == "en":
        """if word_list =[1, 2, 3, 4], return [(1,2), (1,3), (1,4), (2,1), (2,3), (2,4), ...]"""
        tmp = list(product(*[word_list, word_list]))
        return [(x,y) for x, y in tmp if x!=y]
    else:
        print("wrong 'lang' parameter")

def list_combinations(input_list, r):
    combis = list(combinations(input_list, r))
    res = [tuple(combination) for combination in combis]
    return res

def make_similar_lvl_word_pair_list(similar_lvl_word_list):
    """if word_list =[1, 2, 3, 4], return [(1,2), (1,3), (1,4), (2,1), (2,3), (2,4), ...]"""
    # tmp = list(product(*[similar_lvl_word_list, similar_lvl_word_list]))
    return list_combinations(similar_lvl_word_list, 2)

def make_ft_prompt_from_pair(word_pair_list, model="gpt-3.5-turbo", lang="ko", include_similar=False):
    if include_similar:
        if lang == "ko":
            if model == "gpt-3.5-turbo":
                print("making prompts...")

                def make_gpt_style_prompt(word1, word2):

                    res = list()
                    data_line = OrderedDict()
                    data_line["messages"] = [{"role": "system",
                                              "content": "You're a language teaching professional with a keen sense of word level. You are to choose between two words: {word A, word B}. Pick the word you believe is learned first in childhood. When answering: 1.Choose the earlier-learned word. 2. Only state the word, nothing more."
                                              },
                                             {"role": "user",
                                              "content": str(word1) + ", " + str(word2)
                                              },
                                             {"role": "assistant",
                                              "content": str(word1)
                                              },
                                             ]
                    res.append(data_line)
                    data_line = OrderedDict()
                    data_line["messages"] = [{"role": "system",
                                              "content": "You're a language teaching professional with a keen sense of word level. You are to choose between two words: {word A, word B}. Pick the word you believe is learned first in childhood. When answering: 1.Choose the earlier-learned word. 2. Only state the word, nothing more."
                                              },
                                             {"role": "user",
                                              "content": str(word2) + ", " + str(word1)
                                              },
                                             {"role": "assistant",
                                              "content": str(word1)
                                              },
                                             ]
                    res.append(data_line)
                    return res
                res = list()
                for x, y in tqdm(word_pair_list):
                    res += make_gpt_style_prompt(x, y)

                return res

                # with concurrent.futures.ProcessPoolExecutor() as executor:
                #     res = list(tqdm(executor.map(make_gpt_style_prompt,
                #                                  word1_list, word2_list), total=len(word1_list)))
                #     return res
            elif model == "babbage-002":
                print("making prompts... of babbage ...")

                res = list()
                for word1, word2 in tqdm(word_pair_list):
                    if word1 != word2:
                        res.append({
                            "prompt": f"The easier word between '{word1}' and '{word2}' is",
                            "completion": f" '{word1}'."
                        })
                        res.append({
                            "prompt": f"The easier word between '{word2}' and '{word1}' is",
                            "completion": f" '{word1}'."
                        })

                return res

        elif lang == "en":
            if model == "gpt-3.5-turbo":
                print("making prompts...")
                word_variable.df_ku.head()
                def make_gpt_style_prompt(word1, word2):
                    res = list()
                    data_line = OrderedDict()
                    word1_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word1].tolist()[0]
                    word2_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word2].tolist()[0]
                    if word1_lvl < word2_lvl:
                        easier_word = word1
                    else:
                        easier_word = word2
                    data_line["messages"] = [{"role": "system",
                                              "content": "Pick the one word you believe is learned first in childhood."
                                              },
                                             {"role": "user",
                                              "content": str(word1) + ", " + str(word2)
                                              },
                                             {"role": "assistant",
                                              "content": str(easier_word)
                                              },
                                             ]
                    res.append(data_line)

                    return res

                res = list()
                for x, y in tqdm(word_pair_list):
                    res += make_gpt_style_prompt(x, y)

                return res

                # with concurrent.futures.ProcessPoolExecutor() as executor:
                #     res = list(tqdm(executor.map(make_gpt_style_prompt,
                #                                  word1_list, word2_list), total=len(word1_list)))
                #     return res
            elif model == "babbage-002":
                print("making prompts... of babbage ...")

                res = list()
                for word1, word2 in tqdm(word_pair_list):
                    word1_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word1].tolist()[0]
                    word2_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word2].tolist()[0]
                    if word1_lvl < word2_lvl:
                        easier_word = word1
                    else:
                        easier_word = word2
                    if word1 != word2:
                        res.append({
                            "prompt": f"The easier word between '{word1}' and '{word2}' is",
                            "completion": f" '{easier_word}'."
                        })

                return res
    else:
        if lang == "ko":
            if model == "gpt-3.5-turbo":
                print("making prompts...")

                def make_gpt_style_prompt(word1, word2):

                    res = list()
                    data_line = OrderedDict()
                    data_line["messages"] = [{"role": "system",
                                              "content": "You're a language teaching professional with a keen sense of word level. You are to choose between two words: {word A, word B}. Pick the word you believe is learned first in childhood. When answering: 1.Choose the earlier-learned word. 2. Only state the word, nothing more."
                                              },
                                             {"role": "user",
                                              "content": str(word1) + ", " + str(word2)
                                              },
                                             {"role": "assistant",
                                              "content": str(word1)
                                              },
                                             ]
                    res.append(data_line)
                    data_line = OrderedDict()
                    data_line["messages"] = [{"role": "system",
                                              "content": "You're a language teaching professional with a keen sense of word level. You are to choose between two words: {word A, word B}. Pick the word you believe is learned first in childhood. When answering: 1.Choose the earlier-learned word. 2. Only state the word, nothing more."
                                              },
                                             {"role": "user",
                                              "content": str(word2) + ", " + str(word1)
                                              },
                                             {"role": "assistant",
                                              "content": str(word1)
                                              },
                                             ]
                    res.append(data_line)
                    return res
                res = list()
                for x, y in tqdm(word_pair_list):
                    res += make_gpt_style_prompt(x, y)

                return res

                # with concurrent.futures.ProcessPoolExecutor() as executor:
                #     res = list(tqdm(executor.map(make_gpt_style_prompt,
                #                                  word1_list, word2_list), total=len(word1_list)))
                #     return res
            elif model == "babbage-002":
                print("making prompts... of babbage ...")

                res = list()
                for word1, word2 in tqdm(word_pair_list):
                    if word1 != word2:
                        res.append({
                            "prompt": f"The easier word between '{word1}' and '{word2}' is",
                            "completion": f" '{word1}'."
                        })
                        res.append({
                            "prompt": f"The easier word between '{word2}' and '{word1}' is",
                            "completion": f" '{word1}'."
                        })

                return res

        elif lang == "en":
            if model == "gpt-3.5-turbo":
                print("making prompts...")
                word_variable.df_ku.head()
                def make_gpt_style_prompt(word1, word2):
                    res = list()
                    data_line = OrderedDict()
                    word1_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word1].tolist()[0]
                    word2_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word2].tolist()[0]
                    if word1_lvl < word2_lvl:
                        easier_word = word1
                    else:
                        easier_word = word2
                    data_line["messages"] = [{"role": "system",
                                              "content": "Pick the one word you believe is learned first in childhood."
                                              },
                                             {"role": "user",
                                              "content": str(word1) + ", " + str(word2)
                                              },
                                             {"role": "assistant",
                                              "content": str(easier_word)
                                              },
                                             ]
                    res.append(data_line)

                    return res

                res = list()
                for x, y in tqdm(word_pair_list):
                    res += make_gpt_style_prompt(x, y)

                return res

                # with concurrent.futures.ProcessPoolExecutor() as executor:
                #     res = list(tqdm(executor.map(make_gpt_style_prompt,
                #                                  word1_list, word2_list), total=len(word1_list)))
                #     return res
            elif model == "babbage-002":
                print("making prompts... of babbage ...")

                res = list()
                for word1, word2 in tqdm(word_pair_list):
                    word1_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word1].tolist()[0]
                    word2_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word2].tolist()[0]
                    if word1_lvl < word2_lvl:
                        easier_word = word1
                    else:
                        easier_word = word2
                    if word1 != word2:
                        res.append({
                            "prompt": f"The easier word between '{word1}' and '{word2}' is",
                            "completion": f" '{easier_word}'."
                        })

                return res
def make_ft_prompt_from_sim_pair(word_sim_pair_list, model="gpt-3.5-turbo", lang="ko"):

    if lang == "ko":

        if model == "gpt-3.5-turbo":
            print("making prompts...")

            def make_gpt_style_prompt(word1, word2):

                res = list()
                data_line = OrderedDict()
                data_line["messages"] = [{"role": "system",
                                          "content": "You're a language teaching professional with a keen sense of word level. You are to choose between two words: {word A, word B}. Pick the word you believe is learned first in childhood. When answering: 1.Choose the earlier-learned word. 2. Only state the word, nothing more."
                                          },
                                         {"role": "user",
                                          "content": str(word1) + ", " + str(word2)
                                          },
                                         {"role": "assistant",
                                          "content": str(word1)
                                          },
                                         ]
                res.append(data_line)
                data_line = OrderedDict()
                data_line["messages"] = [{"role": "system",
                                          "content": "You're a language teaching professional with a keen sense of word level. You are to choose between two words: {word A, word B}. Pick the word you believe is learned first in childhood. When answering: 1.Choose the earlier-learned word. 2. Only state the word, nothing more."
                                          },
                                         {"role": "user",
                                          "content": str(word2) + ", " + str(word1)
                                          },
                                         {"role": "assistant",
                                          "content": str(word1)
                                          },
                                         ]
                res.append(data_line)
                return res
            res = list()
            for x, y in tqdm(word_sim_pair_list):
                res += make_gpt_style_prompt(x, y)

            return res

            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     res = list(tqdm(executor.map(make_gpt_style_prompt,
            #                                  word1_list, word2_list), total=len(word1_list)))
            #     return res

        elif model == "babbage-002":
            print("making prompts... of babbage ...")

            res = list()
            for word1, word2 in tqdm(word_sim_pair_list):
                if word1 != word2:
                    res.append({
                        "prompt": f"When comparing the ages at which words '{word1}' and '{word2}' are learned, the word learned earlier is",
                        "completion": f" hard to determine."
                    })
                    res.append({
                        "prompt": f"When comparing the ages at which words '{word2}' and '{word1}' are learned, the word learned earlier is",
                        "completion": f" hard to determine."
                    })

            return res

    elif lang == "en":

        if model == "gpt-3.5-turbo":
            print("making prompts...")
            word_variable.df_ku.head()
            def make_gpt_style_prompt(word1, word2):
                res = list()
                data_line = OrderedDict()
                word1_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word1].tolist()[0]
                word2_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word2].tolist()[0]
                if word1_lvl < word2_lvl:
                    easier_word = word1
                else:
                    easier_word = word2
                data_line["messages"] = [{"role": "system",
                                          "content": "Pick the one word you believe is learned first in childhood."
                                          },
                                         {"role": "user",
                                          "content": str(word1) + ", " + str(word2)
                                          },
                                         {"role": "assistant",
                                          "content": str(easier_word)
                                          },
                                         ]
                res.append(data_line)

                return res

            res = list()
            for x, y in tqdm(word_pair_list):
                res += make_gpt_style_prompt(x, y)

            return res

            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     res = list(tqdm(executor.map(make_gpt_style_prompt,
            #                                  word1_list, word2_list), total=len(word1_list)))
            #     return res

        elif model == "babbage-002":
            print("making prompts... of babbage ...")

            res = list()
            for word1, word2 in tqdm(word_pair_list):
                word1_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word1].tolist()[0]
                word2_lvl = word_variable.df_ku['Rating.Mean'][word_variable.df_ku['Word'] == word2].tolist()[0]
                if word1_lvl < word2_lvl:
                    easier_word = word1
                else:
                    easier_word = word2
                if word1 != word2:
                    res.append({
                        "prompt": f"The easier word between '{word1}' and '{word2}' is",
                        "completion": f" '{easier_word}'."
                    })

            return res

def make_wlist_from_word_pair(word_pair_list):
    res = list()
    for word1, word2 in tqdm(word_pair_list):
        res.append(word1)
        res.append(word2)
    return list(set(res))

def save_list_to_jsonl(comp_list, save_file_name="training_data.jsonl"):
    with open(save_file_name, "w", encoding="utf-8") as f:
        for data_line in comp_list:
            json.dump(data_line, f, ensure_ascii=False)  # ensure_ascii로 한글이 깨지지 않게 저장
            f.write("\n")


