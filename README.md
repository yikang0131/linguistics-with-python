# Python语言学应用：从计量到计算
## Course structure
1. Lexical Richness: 1) process BNC corpus (XML format), 2) extract lexical features from BNC, and 3) analyze linguistic data with pandas
2. Syntactic Complexity: 1) introduction to T-unit, 2) pre-process raw corpus, and 3) usage of neosca
3. Readability: 1) introduction to measurements of readability, 2) introduction to coh-metrix, and 3) linguistic features extraction from raw text
4. Sentiment Analysis: 1) corpus building with web scraper, 2) sentiment analysis, and 3) chi-square test
5. Stylometrics: 1) introduction to stylometrics (digital humanities), 2) an example of hierarchical clustering with Yeats' poems
6. Dependency Grammar: 1) introduction to dependency grammar and UD treebanks, 2) process UD treebanks, and 3) analysis of treebanks and visualization
7. Information Entropy: 1) introduction to information theory, and 2) implementation of different algorithms to compute information entropy
8. Word Embeddings: 1) introduction to distributional semantics, and 2) usage of word embeddings
9. Language Model: 1) introduction to principles of language model from N-gram language model to GPT (large pre-trained language model)
10. TBD

## Corpora
- [BNC](http://www.natcorp.ox.ac.uk/): free download, xml annotated
- [WrELFA](https://www.helsinki.fi/en/researchgroups/english-as-a-lingua-franca-in-academic-settings/research/wrelfa-corpus): access via email
- [US presidency archive](https://www.presidency.ucsb.edu/): access via web scraper
- Yeats collections: access via web scraper
- [UD](https://universaldependencies.org/): free download
- [Project Gutenburg](https://www.gutenberg.org/): free dowload
- [Chinese pre-trained word embeddings](https://github.com/Embedding/Chinese-Word-Vectors): free download
- [Glove pre-trained word embeddings](https://nlp.stanford.edu/projects/glove/): free download

## Tools
### Text processing
- ``spaCy``: dependency parsing 
- ``stanza``: preprocess CoNLL format in UD
- ``nltk``: process syntax tree (constituency), n-gram
- ``gensim``: process word embeddings
- ``neosca``: syntactic complexity
- ``StanfordParser``: consistuency parsing (this one is not a Python package)
- ``readability``: readability
- ``textblob``: sentiment analysis

### Data analysis and visualization
- ``pandas``: data manipulation
- ``scipy``: statistics
- ``numpy``: numerical computing
- ``scikit-learn``: data reduction (PCA, TSNE)
- ``matplotlib``: data visualization

### Corpus building
- ``web scraper``: google chrome extension

## References
### Lexical Richness

Shi, Y., & Lei, L. (2021). Lexical use and social class: A study on lexical richness, word length, and word class in spoken English. *Lingua*, 262, 103155.

Lu, X. (2012). The relationship of lexical richness to the quality of ESL learners’ oral narratives. *The Modern Language Journal*, 96(2), 190-208.

Tabari, M. A., Lu, X., & Wang, Y. (2023). The effects of task complexity on lexical complexity in L2 writing: An exploratory study. *System*, 114, 103021.

### Syntactic Complexity

Wu, X., Mauranen, A., & Lei, L. (2020). Syntactic complexity in English as a lingua franca academic writing. *Journal of English for Academic Purposes*, 43, 100798.

Lu, X. (2010). Automatic analysis of syntactic complexity in second language writing. *International journal of corpus linguistics*, 15(4), 474-496.

Hunt, K. W. (1970). Do sentences in the second language grow like those in the first?. *Tesol Quarterly*, 195-202.

### Readability

Crossley, S. A., Greenfield, J., & McNamara, D. S. (2008). Assessing text readability using cognitively based indices. *Tesol Quarterly*, 42(3), 475-493.

Graesser, A. C., McNamara, D. S., Louwerse, M. M., & Cai, Z. (2004). Coh-Metrix: Analysis of text on cohesion and language. *Behavior research methods*, instruments, & computers, 36(2), 193-202.

### Sentiment Analysis

Liu, D., & Lei, L. (2018). The appeal to political sentiment: An analysis of Donald Trump’s and Hillary Clinton’s speech themes and discourse strategies in the 2016 US presidential election. *Discourse, context & media*, 25, 143-152.

### Stylometrics

McIntyre, D., & Walker, B. (2022). Using corpus linguistics to explore the language of poetry: a stylometric approach to Yeats' poems. In *The Routledge Handbook of Corpus Linguistics* (pp. 499-516). Routledge.

Van Hulle, D., & Kestemont, M. (2016). Periodizing Samuel Beckett's works: a stylochronometric approach. *Style*, 50(2), 172-202.

Zhu, H., Lei, L., & Craig, H. (2021). Prose, verse and authorship in Dream of the Red Chamber: A stylometric analysis. *Journal of Quantitative Linguistics*, 28(4), 289-305.

Burrows, J. (2002). ‘Delta’: a measure of stylistic difference and a guide to likely authorship. *Literary and linguistic computing*, 17(3), 267-287.

### Dependency Grammar

De Marneffe, M. C., & Nivre, J. (2019). Dependency grammar. *Annual Review of Linguistics*, 5, 197-218.

De Marneffe, M. C., Dozat, T., Silveira, N., Haverinen, K., Ginter, F., Nivre, J., & Manning, C. D. (2014, May). Universal Stanford dependencies: A cross-linguistic typology. In *LREC* (Vol. 14, pp. 4585-4592).

Liu, H. (2010). Dependency direction as a means of word-order typology: A method based on dependency treebanks. *Lingua*, 120(6), 1567-1578.

Ouyang, J., & Jiang, J. (2018). Can the probability distribution of dependency distance measure language proficiency of second language learners?. *Journal of Quantitative Linguistics*, 25(4), 295-313.

### Information Entropy

Shi, Yaqian, and Lei Lei. Lexical richness and text length: An entropy-based perspective. *Journal of Quantitative Linguistics*, 29(1) (2022): 62-79.

Zhu, H., & Lei, L. (2018). Is modern English becoming less inflectionally diversified? Evidence from entropy-based algorithm. *Lingua*, 216, 10-27.

Zhu, H., & Lei, L. (2018). British cultural complexity: an entropy-based approach. *Journal of Quantitative Linguistics*, 25(2), 190-205.

### Word Embeddings

Boleda, G. (2020). Distributional semantics and linguistic theory. *Annual Review of Linguistics*, 6, 213-234.

Tang, X. (2018). A state-of-the-art of semantic change computation. *Natural Language Engineering*, 24(5), 649-676.

Hu, R., Li, S., & Liang, S. (2019, July). Diachronic sense modeling with deep contextualized word embeddings: An ecological view. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 3899-3908).

### Language Model

Preliminaries on language model

[Stanford cs224n](https://web.stanford.edu/class/cs224n/)

[Speech and language processing](https://web.stanford.edu/~jurafsky/slp3/)
