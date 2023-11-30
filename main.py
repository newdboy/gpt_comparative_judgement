import random

from prepare_ft_data import MakeJsonlForFinetune
import openai_finetune
import openai
import word_variable
from check_accuracy import inference_from_test_pair, calc_accuracy
import pandas as pd
import pickle

# NON-FINETUNE
## Vanilla MODEL (en)
gpt_ko_40 = MakeJsonlForFinetune(model="gpt-3.5-turbo", lang="en", word_num=105,
                                     random_seed=k,
                                     only_high_and_low_lv=False)
train_pair, valid_pair, test_pair = gpt_ko_40.make_pair
train, valid, test = gpt_ko_40.make_prompt_list()
train_dir, valid_dir, test_dir = gpt_ko_40.save_to_jsonl()

df = inference_from_test_pair(test_pair, infer_model="gpt-4",
                              input_token_price_p1k=0.03, output_token_price_p1k=0.06,
                              # input_token_price_p1k=0.0015, output_token_price_p1k=0.002,
                              doubled_inference=False, ft_bool=False)  #if lang='en', doubled_inference=False


## Vanilla MODEL (ko)
gpt_ko_40 = MakeJsonlForFinetune(model="gpt-3.5-turbo", lang="ko", word_num=50,
                                 random_seed=0,
                                 only_high_and_low_lv=False, include_similar=True)
train_pair, valid_pair, test_pair, train_sim_pair, valid_sim_pair, test_sim_pair = gpt_ko_40.make_pair
train, valid, test = gpt_ko_40.make_prompt_list()
train_dir, valid_dir, test_dir = gpt_ko_40.save_to_jsonl()
import random

train_sim_pair

with open("./results/test_pair_common_from50.pickle", 'rb') as file:
    test_pair_common_from50 = pickle.load(file)

df = inference_from_test_pair(test_pair_common_from50, infer_model="gpt-4",
                              input_token_price_p1k=0.03, output_token_price_p1k=0.06,
                              # input_token_price_p1k=0.0015, output_token_price_p1k=0.002,
                              doubled_inference=True, ft_bool=False)  #if lang='en', doubled_inference=False
len(df)
set_dir = "./results/gpt-4_ko_commontest.csv"
df.to_csv(set_dir, encoding='utf-8')
df = pd.read_csv(set_dir)
len(df)
acc = calc_accuracy(df, word_variable.df_kim, lang='ko')
acc

# FINETUNED
## Finetuned MODEL (ko)
gpt_ko_40 = MakeJsonlForFinetune(model="babbage-002", lang="ko", word_num=80,
                                 random_seed=0,
                                 only_high_and_low_lv=False, include_similar=False)
train_pair, valid_pair, test_pair = gpt_ko_40.make_pair
train, valid, test = gpt_ko_40.make_prompt_list()
train_dir, valid_dir, test_dir = gpt_ko_40.save_to_jsonl()


# EVALUATE
df = inference_from_test_pair(test_pair_common_from50, infer_model="ft:babbage-002:pickler:bab-ko-243-epoch5:83B1m0K5",
                              input_token_price_p1k=0.0016, output_token_price_p1k=0.0016,
                              doubled_inference=True, ft_bool=True)  #if lang='en', doubled_inference=False
len(df)
set_dir = "./results/ft_gpt-3.5_ko_34_True_e3.csv"
df.to_csv(set_dir, encoding='utf-8')
df = pd.read_csv(set_dir)
acc = calc_accuracy(df, word_variable.df_kim, lang='ko')
acc

# Finetuned MODEL (en)
gpt_ko_40 = MakeJsonlForFinetune(model="gpt-3.5-turbo", lang="en", word_num=34,
                                 random_seed=0,
                                 only_high_and_low_lv=False)
train_pair, valid_pair, test_pair = gpt_ko_40.make_pair
train, valid, test = gpt_ko_40.make_prompt_list()
train_dir, valid_dir, test_dir = gpt_ko_40.save_to_jsonl()

# ----------------------------------------------------------
# UPLOAD & FINE-TUNE
# ----------------------------------------------------------
# UPLOAD
train_upload_info = openai_finetune.ft_upload_file(train_dir)
valid_upload_info = openai_finetune.ft_upload_file(valid_dir)

train_upload_info['id']
valid_upload_info['id']

# FINE-TUNE
ft_info = openai.FineTuningJob.create(
    training_file=train_upload_info['id'],
    validation_file=valid_upload_info['id'],
    model="babbage-002",
    hyperparameters={
        "n_epochs": 3
    },
    suffix="bab-ko-80_e3_nohl",
)

# Check finetuning status
openai.FineTuningJob.retrieve("ftjob-Z3IIp2DZ8bW49nySgsuXwY7u")

# Cancel a job
openai.FineTuningJob.cancel(ft_info['id'])
openai.FineTuningJob.cancel("ftjob-JttV5qVV98msiF5H9xCc1dZJ")

# Delete a fine-tuned model (must be an owner of the org the model was created in)
openai.Model.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")

train_upload_info['filename']
valid_upload_info['filename']

# curl https://api.openai.com/v1/fine_tuning/jobs \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "training_file": "file-abc123",
#     "validation_file": "file-abc123",
#     "model": "gpt-3.5-turbo",
#     "hyperparameters": {
#       "n_epochs": 1
#     }
#   }'
