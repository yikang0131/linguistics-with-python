import re, os, glob, json
from nltk import sent_tokenize


def clean(text):

    marks = re.findall(r'<.*?>', text, re.S)

    meta_data = [e.replace('<','').replace('>','')
                 for e in marks[0].split('\n') if e]
    
    paper_info = {}
    for info in meta_data:
        k = info.split(':')[0].strip()
        v = info.split(':')[1].strip()
        paper_info[k] = v
    
    text = re.sub(pattern=r'<.*?>', 
                  repl='', 
                  string=text,
                  count=len(marks),
                  flags=re.S)
    
    text = re.sub(pattern=r'\s?\[\d+\]\s?',
                  repl=' ',
                  string=text)
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    lines = [l + '.' if l[-1].isalpha() else l for l in lines ]

    text = ' '.join(lines)
    sentences = sent_tokenize(text)
    sentences = [s + '\n' for s in sentences
                 if len(s) > 4]

    return paper_info, sentences


def preprocess(dir_in, dir_out):
    json_out = dir_out + '/corpus_meta_data.json'
    dir_out += '/' + os.path.basename(dir_in)
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    
    corpus_info = []
    for in_file in glob.glob(f'{dir_in}/*.txt'):
        with open(in_file, 'r', errors='ignore') as fin:
            text = fin.read()
        
        filename = os.path.basename(in_file)
        fp_out = os.path.join(dir_out, filename)

        paper_info, sentences = clean(text)
        with open(fp_out, 'w') as fout:
            fout.writelines(sentences)

        corpus_info.append(paper_info)
    
    with open(json_out, 'w') as json_out:
        json.dump(corpus_info, json_out)


if __name__ == '__main__':
    preprocess('corpus/SciELF', 'tmp/syntax')
        
