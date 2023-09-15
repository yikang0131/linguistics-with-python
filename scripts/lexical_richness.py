import os
import syllables
import pandas as pd
import xml.etree.ElementTree as ET


def extract_soc_and_sent(xml_file):
    '''
        parse a xml file in bnc corpus (xml file should be spoken)

            <person xml:id='' soc=''>...</person>
            </stext>
                <u who=''><s><w hw='' pos=''>text</w></s></u>
            <stext>

        return a list of sentence-social class pair
    '''
    ### check whether the xml is spoken text
    with open(xml_file, 'r', errors='ignore') as fin:
        for _ in range(2):
            line = fin.readline()
            if 'wtext' in line:
                raise ValueError('The script is not spoken materials.')

    root = ET.parse(xml_file).getroot()
    print(xml_file)

    ### get social class of each speaker in one script
    speakers = root.findall('.//person')
    xml_id = '{http://www.w3.org/XML/1998/namespace}id' # xml:id
    speaker2soc = { s.get(xml_id):s.get('soc') for s in speakers }

    ### get sentences and social class of their speakers
    sents_with_soc = []
    utters = root.findall('.//u')

    for utter in utters:
        soc = speaker2soc[utter.get('who')]
        for sent in list(utter):
            words = [(w.text.strip(),w.get('hw'),w.get('c5')) 
                     for w in list(sent) if w.text]
            sents_with_soc.append((words, soc))

    return sents_with_soc


def process_bnc_directory(bnc_root_dir):
    sents_with_soc = []

    for root, dirs, files in os.walk(bnc_root_dir):
        for filename in files:
            file_path = os.path.join(root, filename)

            ### avoid parsing written materials
            try:
                sents_with_soc.extend(extract_soc_and_sent(file_path))
            except:
                continue
    
    sents = [p[0] for p in sents_with_soc]
    soces = [p[1] for p in sents_with_soc]

    tokens, lemmas, poses, soces_ = [],[],[],[]
    for i, s in enumerate(sents):
        tokens.extend([w[0] for w in s])
        lemmas.extend([w[1] for w in s])
        poses.extend([w[2] for w in s])
        soces_.extend([soces[i] for _ in range(len(s))])

    if not os.path.exists('tmp'): os.makedirs('tmp')

    # d = pd.DataFrame({ 'sent': sents, "soc": soces })
    d = pd.DataFrame( {'token': tokens, 'lemma': lemmas, 'pos': poses, 'soc': soces_} )
    for soc in d.soc.unique().tolist():
        split = d[(d.soc==soc)&(~d.pos.isna())&(d.token.str.isalpha())]
        split.to_csv(f'tmp/{soc}.csv', index=False)
    # d.to_csv('tmp/sents_with_soc.csv', index=False)


def compute_total_words(df):
    return len(df)


def compute_ttr(df):
    types_num = len(df.token.str.lower().unique())
    tokens_num = len(df)
    return round(types_num / tokens_num, 4)


def compute_sttr(df, window_size=1000):
    sttr_tmp = []
    for i in range(len(df)):
        if i % 1000:
            continue
        segment = df.iloc[i:i+window_size,]
        sttr_tmp.append(compute_ttr(segment))
    return round(sum(sttr_tmp) / len(sttr_tmp), 4)


def compute_word_length(df):
    mean_wl = round(df.token.str.len().mean(), 4)
    min_wl = int(df.token.str.len().min())
    max_wl = int(df.token.str.len().max())
    std_wl = round(df.token.str.len().std(), 4)
    return { 'mean': mean_wl, 'min': min_wl, 'max': max_wl, 'std': std_wl }


def compute_syllable(df):
    
    def my_estimate(token):
        token = str(token)
        return syllables.estimate(token)

    s = df.token.apply(func=my_estimate)
    mean_s = round(s.mean(), 4)
    min_s = int(s.min())
    max_s = int(s.max())
    std_s = round(s.std(), 4)
    return { 'mean': mean_s, 'min': min_s, 'max': max_s, 'std': std_s }


def get_pos_tag(word_class):
    if word_class == 'NOUN': return 'NN'
    if word_class == 'VERB': return ('VV', 'VH')
    if word_class == 'ADJ': return 'AJ'
    if word_class == 'ADV': return 'AV'
    if word_class == 'PRON': return 'PN'
    if word_class == 'PREP': return 'PR'
    if word_class == 'CONJ': return 'CJ'


def compute_raw_word_class(df, word_class):
    tag = get_pos_tag(word_class)
    if isinstance(tag, tuple):
        return len(df[(df.pos.str[:2]==tag[0])|(df.pos.str[:2]==tag[1])])
    return len(df[df.pos.str[:2]==tag])


def compute_normalised_word_class_(df, word_class, window_size=1000):
    wc_freq_tmp = []
    for i in range(len(df)):
        if i % 1000:
            continue
        segment = df.iloc[i:i+window_size,]
        wc_freq_tmp.append(compute_raw_word_class(segment, word_class))
    return round(sum(wc_freq_tmp) / len(wc_freq_tmp), 4)


def compute_normalised_word_class(df, window_size=1000):
    wc_stats = {}
    for word_class in ['NOUN','VERB','ADJ','ADV','PRON','PREP','CONJ']:
        wc_freq = compute_normalised_word_class_(df, word_class)
        wc_stats[word_class] = wc_freq
    return wc_stats


def get_high_freq_lemmas_(df, word_class, threshold=28.57):
    tag = get_pos_tag(word_class)
    if isinstance(tag, tuple):
        split = (df[(df.pos.str[:2]==tag[0])|(df.pos.str[:2]==tag[1])])
    else:
        split = df[df.pos.str[:2]==tag]
    hf_lemmas = split.lemma.value_counts().apply(func=lambda x: x / len(df) * 10**6)
    return hf_lemmas[hf_lemmas.values>threshold].index.tolist()


def get_high_freq_lemmas(df, threshold=28.57):
    lemmas = {}
    for word_class in ['NOUN','VERB','ADJ','ADV','PRON','PREP','CONJ']:
        hf_lemmas = get_high_freq_lemmas_(df, word_class)
        lemmas[word_class] = hf_lemmas
    return lemmas


def analyze():
    '''
    AB C1 C2 DE
    '''
    high = pd.read_csv('tmp/AB.csv')
    mid = pd.concat([pd.read_csv('tmp/C1.csv'), pd.read_csv('tmp/C2.csv')])
    low = pd.read_csv('tmp/DE.csv')
    dfs = { 'low': low, 'mid': mid, 'high': high }

    results = []
    lemmas = {}
    for soc, df in dfs.items():
        result = [soc]
        total_words = compute_total_words(df)
        ttr = compute_ttr(df)
        sttr = compute_sttr(df)
        wl_stats = compute_word_length(df)
        s_stats = compute_syllable(df)
        wc_stats = compute_normalised_word_class(df)
        hf_lemmas = get_high_freq_lemmas(df)
        result.extend([total_words, ttr, sttr])
        result.extend([v for v in wl_stats.values()])
        result.extend([v for v in s_stats.values()])
        result.extend([v for v in wc_stats.values()])
        results.append(result)
        lemmas[soc] = hf_lemmas

    return results, lemmas


def save(results, lemmas):
    columns = 'soc total_word ttr sttr word_len_mean word_len_min word_len_max word_length_std syl_mean syl_min syl_max syl_std noun_norm_freq verb_norm_freq adj_norm_freq adv_norm_freq pron_norm_freq prep_norm_freq conj_norm_freq'
    pd.DataFrame(data=results, columns=columns.split()).to_csv('tmp/results.csv', index=False)
    if not os.path.exists('tmp/lemmas'): os.makedirs('tmp/lemmas')
    for soc, lemma_ in lemmas.items():
        for word_class, lemma in lemma_.items():
            with open(f'tmp/lemmas/{soc}.{word_class}.txt', 'w') as fout:
                fout.writelines([l+'\n' for l in lemma])


def main():
    # res = extract_soc_and_sent('../data/bnc/Texts/K/KB/KB0.xml')
    # process_bnc_directory('../data/bnc/Texts')
    results, lemmas = analyze()
    # print(results)
    save(results, lemmas)



if __name__ == '__main__':
    main()
    