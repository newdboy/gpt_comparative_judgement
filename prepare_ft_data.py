import random
from typing import List, Any

from sklearn.model_selection import train_test_split
from tool.word_extract_and_save_tool import *
import os

class MakeJsonlForFinetune:
    def __init__(self, model="gpt-3.5-turbo", lang="ko", word_num=30, random_seed=0, only_high_and_low_lv=False, include_similar=True):
        self.model = model
        self.lang = lang
        self.word_num = word_num
        self.random_seed = random_seed
        self.train = list()
        self.test = list()
        self.valid = list()
        self.train_pair = list()
        self.test_pair = list()
        self.valid_pair = list()
        self.train_sim_pair = list()
        self.test_sim_pair = list()
        self.valid_sim_pair = list()
        self.only_high_and_low_lv = only_high_and_low_lv
        self.include_similar = include_similar


        train_expect, test_expect = train_test_split(list(range(self.word_num)), train_size=0.8, test_size=0.2,
                                                     random_state=self.random_seed)
        train_expect, valid_expect = train_test_split(train_expect, train_size=0.75, test_size=0.25,
                                                      random_state=self.random_seed)

        if self.lang == "ko":
            if self.include_similar:
                print("include_similar is True")
                if self.only_high_and_low_lv:
                    train_pair_num_expected = (len(train_expect)**2)*3*2+(len(train_expect)**2-len(train_expect))*3*2
                    valid_pair_num_expected = (len(valid_expect)**2)*3*2+(len(valid_expect)**2-len(valid_expect))*3*2
                    test_pair_num_expected = (len(test_expect)**2)*3*2+(len(test_expect)**2-len(test_expect))*3*2
                    print("train:valid:test is...", train_pair_num_expected, valid_pair_num_expected, test_pair_num_expected)
                else:
                    train_pair_num_expected = (len(train_expect)**2)*3*3*2+(len(train_expect)**2-len(train_expect))*3*3
                    valid_pair_num_expected = (len(valid_expect)**2)*3*3*2+(len(valid_expect)**2-len(valid_expect))*3*3
                    test_pair_num_expected = (len(test_expect)**2)*3*3*2+(len(test_expect)**2-len(test_expect))*3*3
                    print("train:valid:test is...", train_pair_num_expected, valid_pair_num_expected, test_pair_num_expected)

            else:
                if self.only_high_and_low_lv:
                    train_pair_num_expected = (len(train_expect) ** 2) * 3 * 2
                    valid_pair_num_expected = (len(valid_expect) ** 2) * 3 * 2
                    test_pair_num_expected = (len(test_expect) ** 2) * 3 * 2
                    print("train:valid:test is...", train_pair_num_expected, valid_pair_num_expected,
                          test_pair_num_expected)
                else:
                    train_pair_num_expected = (len(train_expect) ** 2) * 3 * 3 * 2
                    valid_pair_num_expected = (len(valid_expect) ** 2) * 3 * 3 * 2
                    test_pair_num_expected = (len(test_expect) ** 2) * 3 * 3 * 2
                    print("train:valid:test is...", train_pair_num_expected, valid_pair_num_expected,
                          test_pair_num_expected)


        elif self.lang == "en":
            print("train:valid:test is...",
                  (len(train_expect) ** 2) - len(train_expect),
                  (len(valid_expect) ** 2) - len(valid_expect),
                  (len(test_expect) ** 2) - len(test_expect))
        else:
            print("wrong 'lang' parameter")

    @property
    def make_pair(self):
        def data_split(wordlist, word_n, seed=0):
            wordlist = extract_random_words_from_list(wordlist, word_n, random_seed=seed)
            train, test = train_test_split(wordlist, train_size=0.8, test_size=0.2, random_state=seed)
            train, validation = train_test_split(train, train_size=0.75, test_size=0.25, random_state=seed)
            return train, validation, test

        if self.lang == "ko":

            if not self.only_high_and_low_lv:
                wlist_12_l_train, wlist_12_l_valid, wlist_12_l_test = data_split(word_variable.kim_12low, word_n=self.word_num)  #seed=self.random_seed 추가
                wlist_12_m_train, wlist_12_m_valid, wlist_12_m_test = data_split(word_variable.kim_12middle, word_n=self.word_num)
                wlist_12_h_train, wlist_12_h_valid, wlist_12_h_test = data_split(word_variable.kim_12high, word_n=self.word_num)

                wlist_34_l_train, wlist_34_l_valid, wlist_34_l_test = data_split(word_variable.kim_34low, word_n=self.word_num)
                wlist_34_m_train, wlist_34_m_valid, wlist_34_m_test = data_split(word_variable.kim_34middle, word_n=self.word_num)
                wlist_34_h_train, wlist_34_h_valid, wlist_34_h_test = data_split(word_variable.kim_34high, word_n=self.word_num)

                wlist_56_l_train, wlist_56_l_valid, wlist_56_l_test = data_split(word_variable.kim_56low, word_n=self.word_num)
                wlist_56_m_train, wlist_56_m_valid, wlist_56_m_test = data_split(word_variable.kim_56middle, word_n=self.word_num)
                wlist_56_h_train, wlist_56_h_valid, wlist_56_h_test = data_split(word_variable.kim_56high, word_n=self.word_num)

                def make_pair_of_splited_data(wlist_12_l, wlist_12_m, wlist_12_h,
                                               wlist_34_l, wlist_34_m, wlist_34_h,
                                               wlist_56_l, wlist_56_m, wlist_56_h, lang="ko"):
                    print("making word pair")
                    wp_list = list()
                    wp_list += make_word_pair_list([wlist_12_l, wlist_12_m], lang=lang)
                    wp_list += make_word_pair_list([wlist_12_l, wlist_12_h], lang=lang)
                    wp_list += make_word_pair_list([wlist_12_m, wlist_12_h], lang=lang)

                    wp_list += make_word_pair_list([wlist_34_l, wlist_34_m], lang=lang)
                    wp_list += make_word_pair_list([wlist_34_l, wlist_34_h], lang=lang)
                    wp_list += make_word_pair_list([wlist_34_m, wlist_34_h], lang=lang)

                    wp_list += make_word_pair_list([wlist_56_l, wlist_56_m], lang=lang)
                    wp_list += make_word_pair_list([wlist_56_l, wlist_56_h], lang=lang)
                    wp_list += make_word_pair_list([wlist_56_m, wlist_56_h], lang=lang)

                    return wp_list

                self.train_pair = make_pair_of_splited_data(wlist_12_l_train, wlist_12_m_train, wlist_12_h_train,
                                                             wlist_34_l_train, wlist_34_m_train, wlist_34_h_train,
                                                             wlist_56_l_train, wlist_56_m_train, wlist_56_h_train,
                                                             lang=self.lang)

                self.test_pair = make_pair_of_splited_data(wlist_12_l_test, wlist_12_m_test, wlist_12_h_test,
                                                           wlist_34_l_test, wlist_34_m_test, wlist_34_h_test,
                                                           wlist_56_l_test, wlist_56_m_test, wlist_56_h_test,
                                                           lang=self.lang)

                self.valid_pair = make_pair_of_splited_data(wlist_12_l_valid, wlist_12_m_valid, wlist_12_h_valid,
                                                            wlist_34_l_valid, wlist_34_m_valid, wlist_34_h_valid,
                                                            wlist_56_l_valid, wlist_56_m_valid, wlist_56_h_valid,
                                                            lang=self.lang)
                if self.include_similar:
                    def make_pair_of_similar(wlist_12_l, wlist_12_m, wlist_12_h,
                                             wlist_34_l, wlist_34_m, wlist_34_h,
                                             wlist_56_l, wlist_56_m, wlist_56_h):
                        wp_list = list()
                        wp_list += make_similar_lvl_word_pair_list(wlist_12_l)
                        wp_list += make_similar_lvl_word_pair_list(wlist_12_m)
                        wp_list += make_similar_lvl_word_pair_list(wlist_12_h)
                        wp_list += make_similar_lvl_word_pair_list(wlist_34_l)
                        wp_list += make_similar_lvl_word_pair_list(wlist_34_m)
                        wp_list += make_similar_lvl_word_pair_list(wlist_34_h)
                        wp_list += make_similar_lvl_word_pair_list(wlist_56_l)
                        wp_list += make_similar_lvl_word_pair_list(wlist_56_m)
                        wp_list += make_similar_lvl_word_pair_list(wlist_56_h)
                        return wp_list

                    self.train_sim_pair = make_pair_of_similar(wlist_12_l_train, wlist_12_m_train, wlist_12_h_train,
                                                               wlist_34_l_train, wlist_34_m_train, wlist_34_h_train,
                                                               wlist_56_l_train, wlist_56_m_train, wlist_56_h_train)

                    self.test_sim_pair = make_pair_of_similar(wlist_12_l_test, wlist_12_m_test, wlist_12_h_test,
                                                              wlist_34_l_test, wlist_34_m_test, wlist_34_h_test,
                                                              wlist_56_l_test, wlist_56_m_test, wlist_56_h_test)

                    self.valid_sim_pair = make_pair_of_similar(wlist_12_l_valid, wlist_12_m_valid, wlist_12_h_valid,
                                                               wlist_34_l_valid, wlist_34_m_valid, wlist_34_h_valid,
                                                               wlist_56_l_valid, wlist_56_m_valid, wlist_56_h_valid)

                    return self.train_pair, self.valid_pair, self.test_pair, self.train_sim_pair, self.valid_sim_pair, self.test_sim_pair
                else:
                    return self.train_pair, self.valid_pair, self.test_pair

            else:
                wlist_12_l_train, wlist_12_l_valid, wlist_12_l_test = data_split(word_variable.kim_12low,
                                                                                 word_n=self.word_num)
                wlist_12_h_train, wlist_12_h_valid, wlist_12_h_test = data_split(word_variable.kim_12high,
                                                                                 word_n=self.word_num)

                wlist_34_l_train, wlist_34_l_valid, wlist_34_l_test = data_split(word_variable.kim_34low,
                                                                                 word_n=self.word_num)
                wlist_34_h_train, wlist_34_h_valid, wlist_34_h_test = data_split(word_variable.kim_34high,
                                                                                 word_n=self.word_num)

                wlist_56_l_train, wlist_56_l_valid, wlist_56_l_test = data_split(word_variable.kim_56low,
                                                                                 word_n=self.word_num)
                wlist_56_h_train, wlist_56_h_valid, wlist_56_h_test = data_split(word_variable.kim_56high,
                                                                                 word_n=self.word_num)

                def make_pair_of_splited_data(wlist_12_l_train, wlist_12_h_train,
                                              wlist_34_l_train, wlist_34_h_train,
                                              wlist_56_l_train, wlist_56_h_train, lang="ko"):
                    wp_list = list()
                    wp_list += make_word_pair_list([wlist_12_l_train, wlist_12_h_train], lang=lang)
                    wp_list += make_word_pair_list([wlist_34_l_train, wlist_34_h_train], lang=lang)
                    wp_list += make_word_pair_list([wlist_56_l_train, wlist_56_h_train], lang=lang)

                    return wp_list

                self.train_pair = make_pair_of_splited_data(wlist_12_l_train, wlist_12_h_train,
                                                       wlist_34_l_train, wlist_34_h_train,
                                                       wlist_56_l_train, wlist_56_h_train,
                                                       lang=self.lang)

                self.test_pair = make_pair_of_splited_data(wlist_12_l_test, wlist_12_h_test,
                                                      wlist_34_l_test, wlist_34_h_test,
                                                      wlist_56_l_test, wlist_56_h_test,
                                                      lang=self.lang)

                self.valid_pair = make_pair_of_splited_data(wlist_12_l_valid, wlist_12_h_valid,
                                                       wlist_34_l_valid, wlist_34_h_valid,
                                                       wlist_56_l_valid, wlist_56_h_valid,
                                                       lang=self.lang)
                if self.include_similar:

                    def make_pair_of_similar(wlist_12_l, wlist_12_h,
                                             wlist_34_l, wlist_34_h,
                                             wlist_56_l, wlist_56_h):
                        wp_list = list()
                        wp_list += make_similar_lvl_word_pair_list(wlist_12_l)
                        wp_list += make_similar_lvl_word_pair_list(wlist_12_h)
                        wp_list += make_similar_lvl_word_pair_list(wlist_34_l)
                        wp_list += make_similar_lvl_word_pair_list(wlist_34_h)
                        wp_list += make_similar_lvl_word_pair_list(wlist_56_l)
                        wp_list += make_similar_lvl_word_pair_list(wlist_56_h)
                        return wp_list

                    self.train_sim_pair = make_pair_of_similar(wlist_12_l_train, wlist_12_h_train,
                                                               wlist_34_l_train, wlist_34_h_train,
                                                               wlist_56_l_train, wlist_56_h_train)

                    self.test_sim_pair = make_pair_of_similar(wlist_12_l_test, wlist_12_h_test,
                                                              wlist_34_l_test, wlist_34_h_test,
                                                              wlist_56_l_test, wlist_56_h_test)

                    self.valid_sim_pair = make_pair_of_similar(wlist_12_l_valid, wlist_12_h_valid,
                                                               wlist_34_l_valid, wlist_34_h_valid,
                                                               wlist_56_l_valid, wlist_56_h_valid)

                    return self.train_pair, self.valid_pair, self.test_pair, self.train_sim_pair, self.valid_sim_pair, self.test_sim_pair

                else:
                    return self.train_pair, self.valid_pair, self.test_pair


        elif self.lang == "en":

            wlist_train, wlist_valid, wlist_test = data_split(word_variable.ku_word_list, word_n=self.word_num)

            self.train_pair = make_word_pair_list(wlist_train, lang=self.lang)
            self.test_pair = make_word_pair_list(wlist_test, lang=self.lang)
            self.valid_pair = make_word_pair_list(wlist_valid, lang=self.lang)

            return self.train_pair, self.valid_pair, self.test_pair
    def make_prompt_list(self):

        if self.include_similar:
            self.train = make_ft_prompt_from_pair(self.train_pair, model=self.model, lang=self.lang, include_similar=True)
            self.test = make_ft_prompt_from_pair(self.test_pair, model=self.model, lang=self.lang, include_similar=True)
            self.valid = make_ft_prompt_from_pair(self.valid_pair, model=self.model, lang=self.lang, include_similar=True)

            self.train += make_ft_prompt_from_sim_pair(self.train_sim_pair, model=self.model, lang=self.lang)
            self.test += make_ft_prompt_from_sim_pair(self.test_sim_pair, model=self.model, lang=self.lang)
            self.valid += make_ft_prompt_from_sim_pair(self.valid_sim_pair, model=self.model, lang=self.lang)

        else:
            self.train = make_ft_prompt_from_pair(self.train_pair, model=self.model, lang=self.lang)
            self.test = make_ft_prompt_from_pair(self.test_pair, model=self.model, lang=self.lang)
            self.valid = make_ft_prompt_from_pair(self.valid_pair, model=self.model, lang=self.lang)

        # shuffle
        random.seed(0)
        self.train = random.sample(self.train, len(self.train))
        self.test = random.sample(self.test, len(self.test))
        self.valid = random.sample(self.valid, len(self.valid))


        print(len(self.train))
        print(len(self.valid))
        print(len(self.test))

        return self.train, self.valid, self.test

    def save_to_jsonl(self):
        def word_num_from_ord_pairs(ord_list, model=self.model):
            """비교 쌍으로 구성된 리스트내에 실제로 단어가 얼마나 다양하게 들어가있는지 단어 수를 계산하기 위한 함수"""
            result = []
            if model == "gpt-3.5-turbo":
                for x in tqdm(ord_list):
                    tmp = x['messages'][1]['content'].split(', ')
                    result += tmp
            elif model == "babbage-002":
                for x in tqdm(ord_list):
                    tmp = x['prompt'].split("'")
                    # print(tmp[1], tmp[3])
                    result.append(tmp[1])
                    result.append(tmp[3])

            return len(list(set(result)))

        def create_directory_if_not_exists(directory_path):
            if not os.path.exists(directory_path):
                os.mkdir(directory_path)
                print(f"Created directory: {directory_path}")
            else:
                print(f"Directory already exists: {directory_path}")

        print("train_word_n, test_word_n, valid_word_n is...", word_num_from_ord_pairs(self.train), word_num_from_ord_pairs(self.test), word_num_from_ord_pairs(self.valid))

        base_dir=(f"./results/ft_{self.model}_"
                  f"{self.lang}_"
                  f"{self.word_num}_"
                  f"ohl({str(self.only_high_and_low_lv)[0]})_"
                  f"sim({str(self.include_similar)[0]})_"
                  f"seed({self.random_seed})/")
        create_directory_if_not_exists(base_dir)

        train_jsonl_dir = f"ft_{self.model}_{self.lang}_{self.word_num}_train.jsonl"
        test_jsonl_dir = f"ft_{self.model}_{self.lang}_{self.word_num}_test.jsonl"
        valid_jsonl_dir = f"ft_{self.model}_{self.lang}_{self.word_num}_valid.jsonl"

        with open(base_dir+f"train{len(self.train)}({word_num_from_ord_pairs(self.train)})_"
                           f"test{len(self.test)}({word_num_from_ord_pairs(self.test)})_"
                           f"valid{len(self.valid)}({word_num_from_ord_pairs(self.valid)}).txt", 'w') as file:
            file.write("")

        save_list_to_jsonl(self.train, save_file_name=base_dir+train_jsonl_dir)
        save_list_to_jsonl(self.test, save_file_name=base_dir+test_jsonl_dir)
        save_list_to_jsonl(self.valid, save_file_name=base_dir+valid_jsonl_dir)

        # save test_pair & easier_word to csv
        # tmp = [(x, y, x) for x,y in self.test_pair]
        # df = pd.DataFrame(tmp, columns=['word1', 'word2', 'easier_word'])
        # df.to_csv(f"./results/{self.model}_{self.lang}_{len(self.test)}_{word_num_from_ord_pairs(self.test)}_test.csv", index=False)

        return base_dir+train_jsonl_dir, base_dir+valid_jsonl_dir, base_dir+test_jsonl_dir

def extract_word_list_from_json_list(ord_list: list)-> list[Any]:
    """
    Extract word list from json list
    To verify test-set isn't contaminated by train-set
    ord_list: list of json format (train, validation, test)
    """
    result = []
    for x in ord_list:
        tmp = x['messages'][1]['content'].split(', ')
        result += tmp
    return result


# EXECUTE IN TERMINAL
# openai tools fine_tunes.prepare_data -f /Users/kintch/PycharmProjects/gpt_comparativ_judgement/results/babbage-002_en_200.jsonl