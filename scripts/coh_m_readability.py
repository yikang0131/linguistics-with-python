import readability

from nltk import FreqDist
from nltk.tree import Tree
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

from os import getenv
from tqdm import tqdm
from neosca.parser import StanfordParser


wnl = WordNetLemmatizer()
stanford_parser_home = getenv('STANFORD_PARSER_HOME')
classpaths = [stanford_parser_home+'/*', stanford_parser_home+'/stanford-tregex.jar']
parser = StanfordParser(classpaths=classpaths)
words = [w.lower() for w in brown.words()]
freq_dist = FreqDist(words)


def sample_text(text):
    lines = text.split('\n')
    length = [len(l) for l in lines]
    sample_idx = length.index(max(length))
    return lines[sample_idx]


def parse_text(sentences, parser):
    tree_strings, trees = [], []
    for sentence in tqdm(sentences):
        parse_result = parser.parse(sentence)
        tree_strings.extend(['(ROOT' + t for t in parse_result.split('(ROOT') if t])
    for tree_string in tree_strings:
        trees.append(Tree.fromstring(tree_string))
    return trees


def get_tree_nodes(tree):
    nodes = []
    for subtree in tree.subtrees(filter=lambda t: t.height() > 2):
        node = [subtree.label()]
        for child in subtree:
            node.append(child.label())
        nodes.append(str(node))
    return nodes


def cal_syntactic_sim(tree_a, tree_b):
    overlap_times = 0
    nodes_a = get_tree_nodes(tree_a)
    nodes_b = get_tree_nodes(tree_b)
    overlap_types = set(nodes_a).intersection(set(nodes_b))
    for node_type in overlap_types:
        overlap_times += nodes_a.count(node_type)
        overlap_times += nodes_b.count(node_type)
    return overlap_times / len(nodes_a + nodes_b)


def lemmatize_sent(sentence):
    lemmas = []

    for word, pos in sentence:
        p = ''
        if pos[0] == 'N': p = 'n'
        if pos[0] == 'V': p = 'v'
        if pos[0] == 'J': p = 'a'
        if pos[0] == 'R': p = 'r'
        
        if p: lemmas.append(wnl.lemmatize(word, p))

    return lemmas


def cal_content_word_overlap(sent_a, sent_b):
    overlap_times = 0
    lemmas_a = lemmatize_sent(sent_a)
    lemmas_b = lemmatize_sent(sent_b)
    overlap_lemmas = set(lemmas_a).intersection(set(lemmas_b))
    for lemma in overlap_lemmas:
        overlap_times += lemmas_a.count(lemma)
        overlap_times += lemmas_b.count(lemma)
    return overlap_times / len(sent_a + sent_b)


def cal_freq(trees):
    freq_list = []
    for tree in trees:
        words = [e[0] for e in tree.pos()]
        for word in words:
            freq_list.append(freq_dist.freq(word.lower()))
    return sum(freq_list) / len(freq_list) * 100


def cal_coh_metrix(sentences, parser):

    trees = parse_text(sentences, parser)
    wfq_total = cal_freq(trees)
    sss_total, cwo_total = 0, 0
    for i in range(len(trees)-1):

        tree_a = trees[i]
        tree_b = trees[i+1]
        sss_total += cal_syntactic_sim(tree_a, tree_b)
        cwo_total += cal_content_word_overlap(tree_a.pos(), tree_b.pos())

    sss_total = sss_total / len(sentences) * 100
    cwo_total = cwo_total / len(sentences) * 100
    return wfq_total, sss_total, cwo_total


def cal_readability(text):
    results = readability.getmeasures(text, lang='en')
    flesch = results['readability grades']['FleschReadingEase']
    kincaid = results['readability grades']['Kincaid']
    return flesch, kincaid


if __name__ == '__main__':
    for i in range(5, 10):
        with open(f'corpus/SciELF/Sci0{i}.txt', 'r', errors='ignore') as fin:
            content = sample_text(fin.read())
        sentences = sent_tokenize(content)
        wfq_total, sss_total, cwo_total = cal_coh_metrix(sentences, parser)
        flesch, kincaid = cal_readability('\n'.join(sentences))
        print('lexical freq:', wfq_total)
        print('syntactic sim:', sss_total)
        print('meaning const:', cwo_total)
        print('flesch ease:', flesch)
        print('kincaid:', kincaid)

