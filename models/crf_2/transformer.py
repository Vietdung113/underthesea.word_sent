from os.path import dirname
from os.path import join

from underthesea.corpus import PlainTextCorpus
from models.crf_2.feature_selection.feature_2 import word2features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


class Transformer:
    def __init__(self):
        self.punctuation = open("punctuation.txt", "r").read().splitlines()

    @staticmethod
    def transform(sentence):
        sentence = [(token,) for token in sentence.split()]
        return sent2features(sentence)

    @staticmethod
    def extract_features(sentence):
        return sent2features(sentence)

    def compound_words(self, token):
        token = token.split('_')
        first_token = [(token[0], "BW")]
        last_token = [(i, "IW") for i in token[1:]]
        return first_token + last_token

    def single_word(self, token):
        if token in self.punctuation:
            return [(token, 'O')]
        else:
            return [(token, 'BW')]

    def tagged_token(self, token):
        if '_' in token:
            return self.compound_words(token)
        else:
            return self.single_word(token)

    def to_column(self, sentence):
        tokens = [token for token in sentence.split()]
        tokens_tagged = [self.tagged_token(token) for token in tokens]
        tokens_tagged = [token_tagged for sub_token_tagged in tokens_tagged for token_tagged in sub_token_tagged]
        return tokens_tagged

    def load_train_sents(self):
        corpus = PlainTextCorpus()
        file_path = join(dirname(dirname(dirname(__file__))), "data", "corpus_2", "train", "input")
        corpus.load(file_path)
        sentences = []
        for document in corpus.documents:
            for sentence in document.sentences:
                if sentence != "":
                    sentences.append(sentence)
        return sentences


def sent2labels(sent):
    return [label for token, label in sent]
