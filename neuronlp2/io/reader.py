__author__ = 'max'

from neuronlp2.io.instance import Sentence, SentenceTree, NERSentence
from neuronlp2.io.common import ROOT, ROOT_POS, ROOT_UPOS, ROOT_CHAR, ROOT_TYPE, ROOT_LEMMA, ROOT_XPOS, ROOT_FEATS, ROOT_DEPS, ROOT_MISC
from neuronlp2.io.common import END, END_POS, END_UPOS, END_CHAR, END_TYPE, END_LEMMA, END_XPOS, END_FEATS, END_DEPS, END_MISC
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH

import numpy as np

class CoNLLXReader(object):
    def __init__(self, file_path):
        self.__source_file = open(file_path, 'r')

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            if not line.startswith('#'): # Attardi
                line = line.strip()
                tokens = line.split('\t')
                lines.append(tokens)
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        char_seqs = []
        lemmas = []
        cpostags = []
        postags = []
        featss = []
        heads = []
        types = []
        depss = []
        miscs = []

        if symbolic_root:
            words.append(ROOT)
            char_seqs.append([ROOT_CHAR])
            lemmas.append(ROOT_LEMMA)
            postags.append(ROOT_POS)
            cpostags.append(ROOT_XPOS)
            featss.append(ROOT_FEATS)
            types.append(ROOT_TYPE)
            heads.append(0)
            depss.append(ROOT_DEPS)
            miscs.append(ROOT_MISC)

        for tokens in lines:
            chars = tokens[1][:MAX_CHAR_LENGTH]
            char_seqs.append(chars)

            word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            lemma = tokens[2]
            cpos = tokens[3]
            pos = tokens[4]
            feats = tokens[5]
            head = int(tokens[6])
            type = tokens[7]
            deps = tokens[8]
            misc = tokens[9]

            words.append(word)
            lemmas.append(lemma)
            postags.append(pos)
            cpostags.append(cpos)
            featss.append(feats)
            heads.append(head)
            types.append(type)
            depss.append(deps)
            miscs.append(misc)

        if symbolic_end:
            words.append(END)
            char_seqs.append([END_CHAR])
            lemmas.append(END_LEMMA)
            cpostags.append(END_XPOS)
            postags.append(END_POS)
            featss.append(END_FEATS)
            heads.append(0)
            types.append(END_TYPE)
            depss.append(END_DEPS)
            miscs.append(END_MISC)

        return SentenceTree(Sentence(words, char_seqs),
                                  lemmas, postags, cpostags,
                                  featss, heads, types,
                                  depss, miscs)


class CoNLLUReader():
    def __init__(self, file_path, normalize_digits=True, symbolic_root=False, symbolic_end=False):
        self.__source_file = open(file_path, 'r')
        self.normalize_digits = normalize_digits
        self.symbolic_root = symbolic_root
        self.symbolic_end = symbolic_end


    def close(self):
        self.__source_file.close()


    def getNext(self):
        words = []
        char_seqs = []
        lemmas = []
        upostags = []
        xpostags = []
        featss = []
        heads = []
        types = []
        depss = []
        miscs = []

        for line in self.__source_file:
            if line.strip() == '': # EOS
                break
            if line.startswith('#'): continue
            tokens = line.split('\t')
            if '-' in tokens[0] or '.' in tokens[0]: # conllu clitics. Attardi
                continue

            word = DIGIT_RE.sub("0", tokens[1]) if self.normalize_digits else tokens[1]
            # trim to MAX_CHAR_LENGTH
            chars = tokens[1][:MAX_CHAR_LENGTH]
            lemma = tokens[2]
            upos = tokens[3]
            xpos = tokens[4]
            feats = tokens[5]
            head = int(tokens[6])
            type = tokens[7]
            deps = tokens[8]
            misc = tokens[9]

            words.append(word)
            char_seqs.append(chars)
            lemmas.append(lemma)
            upostags.append(upos)
            xpostags.append(xpos)
            featss.append(feats)
            heads.append(head)
            types.append(type)
            depss.append(deps)
            miscs.append(misc)

        if not words:
            return None
        
        if self.symbolic_root:
            words.insert(0, ROOT)
            char_seqs.insert(0, [ROOT_CHAR])
            lemmas.insert(0, ROOT_LEMMA)
            upostags.insert(0, ROOT_UPOS)
            xpostags.insert(0, ROOT_XPOS)
            heads.insert(0, 0)
            types.insert(0, ROOT_TYPE)
            depss.insert(0, ROOT_DEPS)
            miscs.insert(0, ROOT_MISC)

        if self.symbolic_end:
            words.append(END)
            char_seqs.append([END_CHAR])
            lemmas.append(END_LEMMA)
            upostags.append(END_UPOS)
            xpostags.append(END_XPOS)
            featss.append(END_FEATS)
            heads.append(0)
            types.append(END_TYPE)
            depss.append(END_DEPS)
            miscs.append(END_MISC)

        return SentenceTree(Sentence(words, char_seqs),
                            lemmas, upostags, xpostags,
                            featss, heads, types, depss, miscs)


class CoNLL03Reader(object):
    def __init__(self, file_path):
        self.__source_file = open(file_path, 'r')

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split(' '))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        words = []
        char_seqs = []
        postags = []
        chunk_tags = []
        ner_tags = []

        for tokens in lines:
            if '-' in tokens[0] or '.' in tokens[0]: # conllu clitics. Attardi
                continue
            chars = []
            for char in tokens[1]:
                chars.append(char)
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)

            word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[2]
            chunk = tokens[3]
            ner = tokens[4]

            words.append(word)
            postags.append(pos)
            chunk_tags.append(chunk)
            ner_tags.append(ner)

        return NERInstance(Sentence(words, char_seqs),
                           postags, chunk_tags, ner_tags)
