from os.path import join, dirname

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from underthesea.corpus import PlainTextCorpus

from pipelines import model_name
from pipelines.data_preparation.to_column import to_column


def get_data():
    output_folder = join(dirname(dirname(dirname(__file__))), "data", "corpus_2", "test", "output")
    model_output_folder = join(dirname(dirname(dirname(__file__))), "data", "corpus_2", "test",
                               "output_%s" % model_name)
    expected_corpus = PlainTextCorpus()
    expected_corpus.load(output_folder)
    actual_corpus = PlainTextCorpus()
    actual_corpus.load(model_output_folder)
    return expected_corpus, actual_corpus


def format_list(list):
    return [i for sub_list in list for i in sub_list]


def get_score():
    expected_corpus, actual_corpus = get_data()
    predict_column = [to_column(sentence) for e in expected_corpus.documents for sentence in e.sentences[:-3]]
    actual_column = [to_column(sentence) for a in actual_corpus.documents for sentence in a.sentences[:-4]]
    predict_column = [format_list(column) for column in predict_column]
    actual_column = [format_list(column) for column in actual_column]
    predict_label = [label[1] for x in predict_column for label in x]
    actual_label = [label[1] for y in actual_column for label in y]
    f1 = f1_score(actual_label, predict_label, list(set(actual_label)), 1, 'weighted', None) * 100
    precision = precision_score(actual_label, predict_label, list(set(actual_label)), 1, 'weighted', None) * 100
    recall = recall_score(actual_label, predict_label, list(set(actual_label)), 1, 'weighted', None) * 100
    print "F1 = %.2f percent" % f1
    print "Precision = %.2f" % precision
    print "Recall = %.2f percent" % recall


if __name__ == '__main__':
    get_score()
