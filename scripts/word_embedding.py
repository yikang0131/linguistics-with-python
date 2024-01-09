import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec


def preview_pretrained_embd(embd_fp):
    with open(embd_fp, 'r') as fin:
        for i, line in enumerate(fin):
            if i == 3: break
            print(line.replace('\n', ''))


def load_pretrained_embd(embd_fp, is_glove=False):
    if is_glove:
        new_embd_fp = embd_fp.replace('.txt', '.w2v.txt')
        if not os.path.exists(new_embd_fp):
            glove2word2vec(embd_fp, new_embd_fp)
        embd_fp = new_embd_fp

    bin_embd_fp = embd_fp.replace('.txt', '.bin')

    if not os.path.exists(bin_embd_fp):
        model = KeyedVectors.load_word2vec_format(embd_fp)
        model.save_word2vec_format(bin_embd_fp)
    else:
        model = KeyedVectors.load_word2vec_format(bin_embd_fp)
    
    return model


def gensim_demo(model):
    # usually use cosine similarity/distance as metrics
    print(model.distance('我','你'))
    print(model.similarity('我','你')) # simlarity = 1 - distance
    print(model.distances('我',['你','他','她','它','他们','动物']))
    print(model.doesnt_match(['你','他','她','它','他们','动物']))
    print(model.most_similar_to_given('猪',['你','他','她','它','他们','动物']))
    print(model.most_similar('你', topn=5))
    # king -> queen: man -> woman    analogy
    # x - man = queen - woman
    # analogy: ? - man = queen - woman / sjtu - No.1 = ? - No.2
    # x = queen + man - woman -> 皇帝
    print(model.most_similar(positive=['皇后','男人'], negative=['女人']))
    print(model.most_similar(positive=['上海交通大学','第二'], negative=['第一']))
    print(model.most_similar(positive=['华中科技大学','第二'], negative=['第一']))
    print(model.most_similar(positive=['交大','第二'], negative=['第一']))


def viz_repr_w_labels(x, y, ofname, is_text=False, texts=None):

    tsne = TSNE(n_components=2, init='random', random_state=42)
    x = tsne.fit_transform(x)

    plt.figure(figsize=(9,12),dpi=200)
    for i, label in enumerate(y.unique().tolist()):
        indices = [i[0] for i in np.argwhere(y.array==label).tolist()]
        plt.scatter(x=x[indices,0],y=x[indices,1],label=label)

        if is_text:
            for i in indices:
                plt.text(x[i,0],x[i,1],texts[i], va='bottom', ha='center', fontsize=8)
    
    plt.legend(fontsize=8)
    plt.savefig(ofname)



