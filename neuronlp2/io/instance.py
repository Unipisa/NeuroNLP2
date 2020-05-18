__author__ = 'max'

__all__ = ['Sentence', 'SentenceTree', 'NERSentence']


class Sentence(object):
    def __init__(self, words, char_seqs):
        self.words = words
        self.char_seqs = char_seqs

    def __len__(self):
        return len(self.words)


class SentenceTree(Sentence):
    def __init__(self, sentence, lemmas, upostags, xpostags, featss, heads, types, depss, miscs):
        super().__init__(sentence.words, sentence.char_seqs)
        self.lemmas = lemmas
        self.upostags = upostags
        self.xpostags = xpostags
        self.featss = featss
        self.heads = heads
        self.types = types
        self.depss = depss
        self.miscs = miscs


class NERSentence(Sentence):
    def __init__(self, sentence, postags, chunk_tags, ner_tags):
        super().__init__(sentence.words, sentence.char_seqs)
        self.postags = postags
        self.chunk_tags = chunk_tags
        self.ner_tags = ner_tags

