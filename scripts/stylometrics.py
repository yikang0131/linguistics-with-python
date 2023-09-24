import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader, stopwords
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler


def get_mfw(corpus_dir, n):
    all_words = []
    works = os.listdir(corpus_dir)
    stopwords = stopwords.words('english')
    for work in works:
        corpus = PlaintextCorpusReader(root=f'{corpus_dir}/{work}', fileids=r'\w+.txt')
        words = [w.lower() for w in corpus.words() if w.isalpha()]
        words = [w for w in words if w not in stopwords]
        all_words.extend(words)

    if not os.path.exists('tmp/stylo'): os.makedirs('tmp/stylo')

    fd = FreqDist(all_words)
    mfw = fd.most_common(n)
    with open(f'tmp/stylo/mfw_{n}', 'w') as fout:
        fout.writelines([f'{w[0]},{w[1]}'+'\n' for w in mfw])
    

def extract_mfw_feat(corpus_dir, n):
    works = os.listdir(corpus_dir)

    if not glob('tmp/stylo/*.txt'): get_mfw(corpus_dir, n)
    mfw_filename = glob('tmp/stylo/*.txt')[0]

    df = pd.read_csv(mfw_filename)
    words = df.iloc[:,0]()[:n]
    all_feats = { 'word':words }
    for work in works:
        corpus = PlaintextCorpusReader(root=f'{corpus_dir}/{work}', fileids=r'\w+.txt')
        fd = FreqDist([w.lower() for w in corpus.words() if w.isalpha()])
        feats = [fd[w] for w in words]
        all_feats[work] = feats
    
    pd.DataFrame(all_feats).to_csv('tmp/stylo/feat.csv', index=False)

    
def to_standard_scalar():
    df = pd.read_csv('tmp/stylo/feat.csv')
    labels = df.columns[1:].tolist()
    labels = [l.split('_')[0] for l in labels]
    feat_mat = df.iloc[:,1:].to_numpy().T
    return StandardScaler().fit_transform(feat_mat)


def viz(feat_mat, labels):
    linkage_matrix = linkage(feat_mat, method='ward')
    plt.figure(figsize=(12, 9))
    dendrogram(linkage_matrix, labels=labels)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.savefig('tmp/stylo/stylo.png')
