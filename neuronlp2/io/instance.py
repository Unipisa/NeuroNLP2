__author__ = 'max'

__all__ = ['Sentence', 'SentenceTree', 'NERInstance']


class Sentence(object):
    def __init__(self, words, word_ids, char_seqs, char_id_seqs):
        self.words = words
        self.word_ids = word_ids
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs

    def __len__(self):
        return len(self.words)


class SentenceTree(Sentence):
    def __init__(self, sentence, lemmas, postags, pos_ids, xpostags, xpos_ids, featss, heads, types, type_ids, depss, miscs):
        self.__super__(Sentence, sentence.words, sentence.word_ids,
                       sentence.char_seqs, sentence.char_id_seqs)
        self.lemmas = lemmas
        self.postags = postags
        self.pos_ids = pos_ids
        self.xpostags = xpostags
        self.xpos_ids = xpos_ids
        self.featss = featss
        self.heads = heads
        self.types = types
        self.type_ids = type_ids
        self.depss = depss
        self.miscs = miscs


class NERInstance(object):
    def __init__(self, sentence, postags, pos_ids, chunk_tags, chunk_ids, ner_tags, ner_ids):
        self.sentence = sentence
        self.postags = postags
        self.pos_ids = pos_ids
        self.chunk_tags = chunk_tags
        self.chunk_ids = chunk_ids
        self.ner_tags = ner_tags
        self.ner_ids = ner_ids

    def length(self):
        return self.sentence.length()
