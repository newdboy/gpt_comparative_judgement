import pandas as pd


# Kim et al.(2021)'s list
df_kim = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/김한샘어휘_20220902.xlsx')
df_kim.head()
df_dd = df_kim.drop_duplicates(subset=['단어'])
df_kim = df_dd[:]
kim_word_list = df_dd['단어'].tolist()

# 1-2학년
kim_12high = df_dd['단어'][(df_dd.학년군 == '1~2') & (df_dd.수준별등급 == '상')].tolist()
kim_12middle = df_dd['단어'][(df_dd.학년군 == '1~2') & (df_dd.수준별등급 == '중')].tolist()
kim_12low = df_dd['단어'][(df_dd.학년군 == '1~2') & (df_dd.수준별등급 == '하')].tolist()

# 3-4학년
kim_34high = df_dd['단어'][(df_dd.학년군 == '3~4') & (df_dd.수준별등급 == '상')].tolist()
kim_34middle = df_dd['단어'][(df_dd.학년군 == '3~4') & (df_dd.수준별등급 == '중')].tolist()
kim_34low = df_dd['단어'][(df_dd.학년군 == '3~4') & (df_dd.수준별등급 == '하')].tolist()

# 5-6학년
kim_56high = df_dd['단어'][(df_dd.학년군 == '5~6') & (df_dd.수준별등급 == '상')].tolist()
kim_56middle = df_dd['단어'][(df_dd.학년군 == '5~6') & (df_dd.수준별등급 == '중')].tolist()
kim_56low = df_dd['단어'][(df_dd.학년군 == '5~6') & (df_dd.수준별등급 == '하')].tolist()

## Kuperman's list
df_ku = pd.read_excel('/Users/kintch/Library/CloudStorage/Dropbox/sj/2023-2/연구/[진행중]어휘 난도_chatgpt/data for evaluation/AoA_ratings_Kuperman_et_al_BRM.xlsx')
df_ku.head()
ku_word_list = df_ku.Word.tolist()
