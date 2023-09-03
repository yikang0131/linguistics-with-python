import os, sys
import numpy as np
import pandas as pd
from stanza.utils.conll import CoNLL
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--ud_root_dir', dest='ud_root_dir')
    parser.add_argument('--lang', dest='lang', nargs='+')
    parser.add_argument('--output_dir', dest='output_dir', default='tmp')
    args = parser.parse_args()
    valid_langs = [l.replace('UD_', '') 
                   for l in os.listdir(args.ud_root_dir)] 
    for l in args.lang:
        if l not in valid_langs:
            raise ValueError (f'There is no {l}.')
    langs = tuple(args.lang)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_csv_path = os.path.join(args.output_dir, 'result.csv')
    process_many_langs(args.ud_root_dir, args.output_dir, output_csv_path, *langs)
    

def load_treebanks(lang, ud_root_dir):
    '''
        load all conllus file of the given language
        and return a concatenated dataframe.

        lang: the code for language used in UD
            such as "Chinese-GSDSimp"
        ud_root_dir: the root directory of UD treebanks
    '''
    ud_dir = os.path.join(ud_root_dir, f'UD_{lang}')
    conllu_fs = [os.path.join(ud_dir, f) 
                 for f in os.listdir(ud_dir)
                 if os.path.splitext(f)[1] == '.conllu']
    
    files_num = len(conllu_fs)
    ds = [conll2dataframe(c_f) for c_f in conllu_fs]
    d = pd.concat(ds)
    sents_num = len(d.sent_id.unique())
    tokens_num = len(d)
    lang_info_pat = '{}\t#files: {}\t#sents: {}\t#tokens: {}'
    print(lang_info_pat.format(lang, files_num, sents_num, tokens_num))

    return d


def conll2dataframe(conllu_f):
    '''load a conllu file and convert to dataframe'''
    conll_list = []
    conll_dict = CoNLL.conll2dict(conllu_f)
    sents, texts = conll_dict[0], conll_dict[1]
    for i, sent in enumerate(sents):
        # '#sent_id = train-s1'
        for meta_data in texts[i]:
            if 'sent_id' in meta_data:
                sent_id = meta_data.split('=')[-1].strip()
        for token in sent:
            token['sent_id'] = sent_id
            # 'id' = (1,)
            token['id'] = token['id'][0]
            conll_list.append(token)  
    d = pd.DataFrame(conll_list)
    return d[~d.deprel.isna()]


def analyze_one_lang(df):
    '''
        analyze one language. in this demo, 
        we analyze (1) dependency relations,
        and (2) mean dependency distance (MDD)

        return a dictionary structured as:
        {
          'advmod': {'perc': 12.12 'md': 1.2112},
          'root': {'perc': 9.11, 'md': 0}
        }
    '''
    all_dep_rels = df.deprel.unique().tolist()
    res = {dp: {'perc': 0, 'mdd': 0} for dp in all_dep_rels}
    for dep_rel in all_dep_rels:
        sub_d = df[df.deprel==dep_rel]
        perc = round(len(sub_d) / len(df), 4) * 100
        res[dep_rel]['perc'] = perc
        mdd = np.average(np.abs(sub_d['id']-sub_d['head']))
        # tmp = 0
        # for _, row in sub_d.iterrows():
        #     tmp += abs(row['id'] - row['head'])
        # mdd = tmp / len(sub_d)
        res[dep_rel]['mdd'] = round(mdd, 2)
    return res


def cmp_across_langs(result_dict, output_csv):
    '''
        get common dependency relations across langauges
        store the comparable results into a csv file

        dep_rel feat    lang1   lang2   ... langn
        advmod  mdd     1.13    1.12    ... 1.12
        ...     mdd     
        advmod  perc    12.12   13.12   ... 14.12
        ...     perc
    '''
    langs = result_dict.keys()
    common_dep_rels = list(set.intersection(
        *(set(result_dict[l].keys()) for l in langs)))
    
    output_result = {'dep_rel': common_dep_rels * 2,
                     'feat': ['mdd'] * len(common_dep_rels) + 
                             ['perc'] * len(common_dep_rels)}
    for lang in langs:
        output_result[lang] = []
    for dep_rel in common_dep_rels:
        for lang in langs:
            output_result[lang].append(
                result_dict[lang][dep_rel]['mdd'])
    for dep_rel in common_dep_rels:
        for lang in langs:
            output_result[lang].append(
                result_dict[lang][dep_rel]['perc'])
    output_df = pd.DataFrame(output_result)
    output_df.to_csv(output_csv, index=False)


def visualize(output_csv, output_dir):
    output_df = pd.read_csv(output_csv)
    feats = output_df.feat.unique().tolist()
    for feat in feats:
        sub_d = output_df[output_df.feat==feat]
        ax = sub_d.plot.bar(x='dep_rel')
        ax.get_figure().savefig(f'{output_dir}/{feat}.png')


def process_many_langs(ud_root_dir, output_dir, csv_path, *args):
    result_dict = {}
    for lang in args:
        treebank = load_treebanks(lang, ud_root_dir)
        result_dict[lang] = analyze_one_lang(treebank)
    cmp_across_langs(result_dict, csv_path)
    visualize(csv_path, output_dir)


def demo():
    process_many_langs('ud-treebanks-v2.12', 'tmp/result.csv',
                       'Chinese-GSDSimp', 'English-EWT',
                       'French-GSD', 'Japanese-GSD')


if __name__ == '__main__':
    main()
