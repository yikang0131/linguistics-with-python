from nltk.corpus import reuters
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline


lm = MLE(3) # n-gram: increasing n -> more computing cost
train, vocab = padded_everygram_pipeline(3, reuters.sents())
lm.fit(train, vocab)
lm.generate(text_seed=['I', 'don', '\'t', 'want', 'to', 'have'], num_words=5)
