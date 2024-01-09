import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader, stopwords
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WORK_INFO = {'cross': '1889', 'rose': '1893', 'wind': '1899', 'seven': '1904', 'helmet': '1910', 'respons': '1914', 'swans': '1919', 'robartes': '1921', 'tower': '1928', 'stairs': '1933', 'parnell': '1935', 'new': '1938', 'last': 1939}


def get_mfw(corpus_dir, n):
    all_words = []
    works = os.listdir(corpus_dir)
    stopwords_en = stopwords.words('english')
    for work in works:
        if work == 'narrative':
            continue
        corpus = PlaintextCorpusReader(root=f'{corpus_dir}/{work}', fileids=r'\w+.txt')
        words = [w.lower() for w in corpus.words() if w.isalpha()]
        words = [w for w in words if w not in stopwords_en]
        all_words.extend(words)

    if not os.path.exists('tmp/stylo'): os.makedirs('tmp/stylo')

    fd = FreqDist(all_words)
    mfw = fd.most_common(n)
    with open(f'tmp/stylo/mfw_{n}.txt', 'w') as fout:
        lines = ['word,freq\n'] + [f'{w[0]},{w[1]}'+'\n' for w in mfw]
        fout.writelines(lines)
    

def extract_mfw_feat(corpus_dir, n):
    works = os.listdir(corpus_dir)

    if not glob(f'tmp/stylo/mfw_{n}.txt'): get_mfw(corpus_dir, n)
    mfw_filename = glob(f'tmp/stylo/mfw_{n}.txt')[0]

    df = pd.read_csv(mfw_filename)
    words = df.iloc[:,0][:n]
    all_feats = { 'word':words }
    for work in works:
        if work == 'narrative':
            continue
        corpus = PlaintextCorpusReader(root=f'{corpus_dir}/{work}', fileids=r'\w+.txt')
        fd = FreqDist([w.lower() for w in corpus.words() if w.isalpha()])
        feats = [fd[w] for w in words]
        all_feats[work] = feats

    pd.DataFrame(all_feats).to_csv('tmp/stylo/feat.csv', index=False)

    
def to_pca(df):
    # df = pd.read_csv(csv_filepath)
    works = df.columns.tolist()[1:]
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df.T[1:])

    plt.figure(figsize=(12, 9))
    for i, data in enumerate(principal_components):
        plt.scatter(data[0], data[1])
        plt.annotate(WORK_INFO[works[i]], (data[0], data[1]))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization')
    plt.savefig('tmp/stylo/pca.png')


def to_dendrogram(df):
    # df = pd.read_csv(csv_filepath)
    works = df.columns.tolist()[1:]
    labels = [WORK_INFO[w] for w in works]
    # feat_mat = df.iloc[:,1:].to_numpy().T
    scaler = StandardScaler()
    data = df.T[1:].to_numpy()
    scaler.fit(data)
    feat_mat = scaler.transform(data)
    linkage_matrix = linkage(feat_mat, method='ward')

    plt.figure(figsize=(12, 9))
    dendrogram(linkage_matrix, labels=labels)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.savefig('tmp/stylo/dendrogram.png')


if __name__ == '__main__':
    mfw_n = 100
    get_mfw('yeats', mfw_n)
    extract_mfw_feat('yeats', mfw_n)
    df = pd.read_csv('tmp/stylo/feat.csv')
    to_pca(df)
    to_dendrogram(df)
