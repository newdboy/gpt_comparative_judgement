import sys
import google.protobuf.text_format as tf
from bareunpy import Tagger

#
# you can API-KEY from https://bareun.ai/
#
API_KEY="koba-C7OMRNQ-2S3UUDQ-QSNOXFQ-KEZIWDI"

# If you have your own localhost bareun.
my_tagger = Tagger(API_KEY, 'localhost')
# or if you have your own bareun which is running on 10.8.3.211:15656.
# my_tagger = Tagger(API_KEY, '10.8.3.211', 15656)


# print results.
res = my_tagger.tags(["안녕하세요.", "반가워요!"])
para = '''사람들이 지속적으로 책을 읽는 이유 중 하나는 즐거움이다. 독서의 즐거움에는 여러 가지가 있겠지만 그 중심에는 ‘소통의 즐거움’이 있다.
독자는 독서를 통해 책과 소통하는 즐거움을 경험한다. 독서는 필자와 간접적으로 대화하는 소통 행위이다. 독자는 자신이 속한 사회나 시대의 영향 아래 필자가 속해 있거나 드러내고자 하는 사회나 시대를 경험한다. 직접 경험하지 못했던 다양한 삶을 필자를 매개로 만나고 이해하면서 독자는 더 넓은 시야로 세계를 바라볼 수 있다. 이때 같은 책을 읽은 독자라도 독자의 배경 지식이나 관점 등의 독자 요인, 읽기 환경이나 과제 등의 상황 요인이 다르므로, 필자가 보여 주는 세계를 그대로 수용하지 않고 저마다 소통 과정에서 다른 의미를 구성할 수 있다.
이러한 소통은 독자가 책의 내용에 대해 질문하고 답을 찾아내는 과정에서 가능해진다. 독자는 책에서 답을 찾는 질문, 독자 자신에게서 답을 찾는 질문 등을 제기할 수 있다.
전자의 경우 책에 명시된 내용에서 답을 발견할 수 있고, 책의 내용들을 관계 지으며 답에 해당하는 내용을 스스로 구성할 수도 있다. 또한 후자의 경우 책에는 없는 독자의 경험에서 답을 찾을 수 있다. 이런 질문들을 풍부히 생성하고 주체적으로 답을 찾을 때 소통의 즐거움은 더 커진다.
'''
res = my_tagger.morphs(para)
res
# get protobuf message.
m = res.msg()
tf.PrintMessage(m, out=sys.stdout, as_utf8=True)
print(tf.MessageToString(m, as_utf8=True))
print(f'length of sentences is {len(m.sentences)}')
## output : 2
print(f'length of tokens in sentences[0] is {len(m.sentences[0].tokens)}')
print(f'length of morphemes of first token in sentences[0] is {len(m.sentences[0].tokens[0].morphemes)}')
print(f'lemma of first token in sentences[0] is {m.sentences[0].tokens[0].lemma}')
print(f'first morph of first token in sentences[0] is {m.sentences[0].tokens[0].morphemes[0]}')
print(f'tag of first morph of first token in sentences[0] is {m.sentences[0].tokens[0].morphemes[0].tag}')
# get json object
from pprint import pprint
jo = res.as_json()
pprint(jo)

# 여기서부터 시작해야 할 듯.
pprint(jo)

# get tuple of pos tagging.
pa = res.pos()
print(pa)
# another methods
ma = res.morphs()
print(ma)
na = res.nouns()
print(na)
va = res.verbs()
print(va)