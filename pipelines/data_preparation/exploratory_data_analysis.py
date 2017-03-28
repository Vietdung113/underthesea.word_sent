from os.path import dirname, join

import pandas as pd
from underthesea.corpus import PlainTextCorpus


def count_token(documents):
    count = 0
    for document in documents:
        for sentences in document.sentences:
            for word in sentences.split(' '):
                count += 1
    return count


f = open(join(dirname(dirname(__file__)), "reports", "exploratory_data_analysis.txt"), "w")
f.write("[Statistics] Train Data Set\n")
train_folder = join(dirname(dirname(dirname(__file__))), "data", "corpus_2", "train", "input")
train_corpus = PlainTextCorpus()
train_corpus.load(train_folder)
# (new_word, word_in_dictionary) = compare_dictionary(train_folder)
f.write("Total documents: %d\n" % len(train_corpus.documents))
s = pd.Series([len(d.sentences) for d in train_corpus.documents])
sentences = [sentence for document in train_corpus.documents for sentence in document.sentences]

print("total sentence train : %s" % len(sentences))
print(pd.Series.describe(s))

f.write("Min token in sentence %d\n" % s.describe()['min'])
f.write("Max token in sentence %d\n" % s.describe()['max'])
f.write("Total sentences: %d\n" % sum(s))
f.write("Total token: %d\n" % count_token(train_corpus.documents))
# f.write("New word :%d \n" % len(new_word))
# f.write("Word in dictionary:git  %d\n" % len(word_in_dictionary))
# coverage = float(len(new_word)) / float(len(viet_dict_11K.words))
# f.write("Word coverage: %0.2f\n" % coverage)
f.write("\n")

f.write("[Statistics] Test Data Set\n")
train_folder = join(dirname(dirname(dirname(__file__))), "data", "corpus_2", "test", "output")
test_corpus = PlainTextCorpus()
test_corpus.load(train_folder)
# (new_word, word_in_dictionary) = compare_dictionary(train_folder)

s = pd.Series([len(d.sentences) for d in test_corpus.documents])
sentences = [sentence for document in test_corpus.documents for sentence in document.sentences]
print ("total sentence test: %d" % len(sentences))
print (pd.Series.describe(s))

f.write("Total documents: %d\n" % len(test_corpus.documents))
f.write("Min token in sentence %d\n" % s.describe()['min'])
f.write("Max token in sentence %d\n" % s.describe()['max'])
f.write("Total sentences: %d\n" % sum(s))
f.write("Total token: %d\n" % count_token(test_corpus.documents))
# f.write("New word :%d \n" % len(new_word))
# f.write("Word in dictionary: %d\n" % len(word_in_dictionary))
# coverage = float(len(new_word)) / float(len(viet_dict_11K.words))
# f.write("Word coverage: %0.2f" % coverage)

f.write("\n")
