import os, random, math
import numpy as np

from nltk import FreqDist
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup


class BookForEntropy(object):

    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        self.wdw_sizes = [ 50,100,200,300,800,1000 ]
        while self.total_word_num > self.wdw_sizes[-1]:
            self.wdw_sizes.append(self.wdw_sizes[-1]+1000)

    @classmethod
    def from_html(cls, html_path):
        with open(html_path, 'r', encoding='utf-8') as fin:
            html = BeautifulSoup(fin.read(), 'html.parser')
        tokenized_paras = [word_tokenize(p.get_text().\
            lower().replace('\n', ' ')) for p in html.find_all('p')]
        return cls(tokenized_paras)

    @property
    def total_word_num(self):
        return sum([len(p) for p in self.paragraphs])

    def randomize(self, seed=42):
        random.seed(seed)
        random.shuffle(self.paragraphs)

    def slice(self, window_size):
        words = []
        for paragraph in self.paragraphs:
            if len(words) > window_size:
                break
            words.extend(paragraph)
        slice = words[:window_size]
        return len(slice), slice


def naive_shannon_entropy(slice):
    fd = FreqDist(slice)
    entropy = 0
    for word in fd:
        p = fd.freq(word)
        entropy -= p * math.log2(p)
    return entropy


def zhang_shannon_entropy(slice, mode='fast'):
    if mode == 'fast':
        return fast_zhang(slice)
    fd = FreqDist(slice)
    T = fd.N()
    entropy = 0
    for word in fd:
        pi = fd.freq(word)
        fi = fd[word]
        s_tmp = 0
        for v in range(1, T-fi+1):
            m_tmp = 1
            for j in range(0, v):
                m_tmp *= 1 + (1 - fi) / (T - 1 - j)
            s_tmp += m_tmp / v
        entropy += s_tmp * pi
    return entropy


def fast_zhang(slice):
    fd = FreqDist(slice)
    T = fd.N()
    entropy = 0
    f_dict, v_dict = {}, {}

    '''
    R(v, f) = k(f) * R(v-f, f): dynamic planning
    '''
    def R(v, f):
        if v in v_dict[f]:
            return v_dict[f][v]
        if not v in v_dict[f]:
            if v == 0:
                v_dict[f][v] = 1
                return v_dict[f][v]
            v_dict[f][v] = (1+(1-f)/(T-v)) * R(v-1, f)       
        return v_dict[f][v]

    def Q(f):
        q = np.sum([R(v, f) / v 
            for v in range(1, T-f+1)])
        return q
    
    for word in fd:
        pi = fd.freq(word)
        fi = fd[word]

        if not fi in v_dict:
            v_dict[fi] = {}
        
        if not fi in f_dict:
            f_dict[fi] = Q(fi)

        entropy += f_dict[fi] * pi

    return entropy


def main():
    persuasion = BookForEntropy.from_html(
        'data/austen/raw/persuasion.utf8')
    for _ in range(1): # times for randomize
        persuasion.randomize()
        for i, wdw_size in enumerate(persuasion.wdw_sizes[:50]):
            word_n, slice = persuasion.slice(wdw_size)
            z1 = naive_shannon_entropy(slice)
            z2 = zhang_shannon_entropy(slice)
            print(word_n, z1, z2)


if __name__ == '__main__':
    main()
