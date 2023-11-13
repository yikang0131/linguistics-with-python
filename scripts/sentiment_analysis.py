from nrclex import NRCLex
from nltk.corpus import PlaintextCorpusReader
from scipy.stats import chi2_contingency


def compute_positive_sentiment(corpus, fileids):
    cnt = 0
    for i, sent in enumerate(corpus.sents(fileids)):
        # blob = TextBlob(' '.join(sent))
        emotion = NRCLex(' '.join(sent))
        if emotion.affect_frequencies['positive'] < emotion.affect_frequencies['negative']:
            cnt += 1
    return cnt, i + 1


def chi_square_test(observed):
    chi2_stat, p_value, dof, _ = chi2_contingency(observed)
    return chi2_stat, p_value, dof


if __name__ == '__main__':
    trump = PlaintextCorpusReader('data/presidency/trump', fileids='.*\.txt')
    clinton = PlaintextCorpusReader('data/presidency/hilary', fileids='.*\.txt')
    t_pos, t_all = compute_positive_sentiment(trump, trump.fileids())
    c_pos, c_all = compute_positive_sentiment(clinton, clinton.fileids())
    print(chi_square_test([[t_pos, t_all - t_pos], [c_pos, c_all - c_pos]]))


