__author__ = 'max'

from neuronlp2.io.instance import Sentence, SentenceTree, NERInstance
from neuronlp2.io.common import ROOT, ROOT_POS, ROOT_CHAR, ROOT_TYPE, ROOT_LEMMA, ROOT_XPOS, ROOT_FEATS, ROOT_DEPS, ROOT_MISC
from neuronlp2.io.common import END, END_POS, END_CHAR, END_TYPE, END_LEMMA, END_XPOS, END_FEATS, END_DEPS, END_MISC
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH

import numpy as np

class CoNLLXReader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__cpos_alphabet = cpos_alphabet
        self.__type_alphabet = type_alphabet

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
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        lemmas = []
        cpostags = []
        postags = []
        pos_ids = []
        featss = []
        heads = []
        types = []
        type_ids = []
        depss = []
        miscs = []

        if symbolic_root:
            words.append(ROOT)
            word_ids.append(self.__word_alphabet.get_index(ROOT))
            char_seqs.append([ROOT_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(ROOT_CHAR), ])
            lemmas.append(ROOT_LEMMA)
            postags.append(ROOT_POS)
            pos_ids.append(self.__pos_alphabet.get_index(ROOT_POS))
            cpostags.append(ROOT_XPOS)
            cpos_ids.append(self.__cpos_alphabet.get_index(ROOT_XPOS))
            featss.append(ROOT_FEATS)
            types.append(ROOT_TYPE)
            type_ids.append(self.__type_alphabet.get_index(ROOT_TYPE))
            heads.append(0)
            depss.append(ROOT_DEPS)
            miscs.append(ROOT_MISC)

        for tokens in lines:
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

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
            word_ids.append(self.__word_alphabet.get_index(word))

            lemmas.append(lemma)
            
            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            cpostags.append(cpos)
            cpos_ids.append(self.__cpos_alphabet.get_index(cpos))

            featss.append(feats)

            heads.append(head)

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            depss.append(deps)
            miscs.append(misc)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            lemmas.append(END_LEMMA)
            cpostags.append(END_XPOS)
            cpos_ids.append(self.__cpos_alphabet.get_index(END_CPOS))
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            featss.append(END_FEATS)
            heads.append(0)
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            depss.append(END_DEPS)
            miscs.append(END_MISC)

        return SentenceTree(Sentence(words, word_ids, char_seqs, char_id_seqs),
                                  np.array(lemmas), postags, pos_ids, cpostags, cpos_ids,
                                  np.array(featss), heads, types, type_ids,
                                  np.array(depss), np.array(miscs))


class CoNLLUReader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, xpos_alphabet, type_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__xpos_alphabet = xpos_alphabet
        self.__type_alphabet = type_alphabet

    def close(self):
        self.__source_file.close()

    def getNext(self, normalize_digits=True, symbolic_root=False, symbolic_end=False):

        words = []
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        lemmas = []
        postags = []
        pos_ids = []
        xpostags = []
        xpos_ids = []
        featss = []
        heads = []
        types = []
        type_ids = []
        depss = []
        miscs = []

        for line in self.__source_file:
            if line.strip() == '': # EOS
                break
            if line.startswith('#'): continue
            tokens = line.split('\t')
            if '-' in tokens[0] or tokens[0].startswith('.'): # conllu clitics. Attardi
                continue
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            lemma = tokens[2]
            pos = tokens[3]
            xpos = tokens[4]
            feats = tokens[5]
            head = int(tokens[6])
            type = tokens[7]
            deps = tokens[8]
            misc = tokens[9]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            lemmas.append(lemma)

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            xpostags.append(xpos)
            xpos_ids.append(self.__xpos_alphabet.get_index(xpos))
            featss.append(feats)

            heads.append(head)

            types.append(type)
            type_ids.append(self.__type_alphabet.get_index(type))

            depss.append(deps)
            miscs.append(misc)

        if not words:
            return None
        
        if symbolic_root:
            words.insert(ROOT, 0)
            word_ids.insert(self.__word_alphabet.get_index(ROOT), 0)
            char_seqs.insert([ROOT_CHAR, ], 0)
            char_id_seqs.insert([self.__char_alphabet.get_index(ROOT_CHAR), ], 0)
            lemmas.insert(ROOT_LEMMA, 0)
            postags.insert(ROOT_POS, 0)
            pos_ids.insert(self.__pos_alphabet.get_index(ROOT_POS), 0)
            xpostags.insert(ROOT_XPOS, 0)
            xpos_ids.insert(self.__xpos_alphabet.get_index(ROOT_XPOS), 0)
            heads.insert(0, 0)
            types.insert(ROOT_TYPE, 0)
            type_ids.insert(self.__type_alphabet.get_index(ROOT_TYPE), 0)
            depss.insert(ROOT_DEPS, 0)
            miscs.insert(ROOT_MISC, 0)

        if symbolic_end:
            words.append(END)
            word_ids.append(self.__word_alphabet.get_index(END))
            char_seqs.append([END_CHAR, ])
            char_id_seqs.append([self.__char_alphabet.get_index(END_CHAR), ])
            lemmas.append(END_LEMMA)
            postags.append(END_POS)
            pos_ids.append(self.__pos_alphabet.get_index(END_POS))
            xpostags.append(END_XPOS)
            xpos_ids.append(self.__xpos_alphabet.get_index(END_XPOS))
            featss.append(END_FEATS)
            heads.append(0)
            types.append(END_TYPE)
            type_ids.append(self.__type_alphabet.get_index(END_TYPE))
            depss.append(END_DEPS)
            miscs.append(END_MISC)

        return SentenceTree(Sentence(words, word_ids, char_seqs, char_id_seqs),
                                  np.array(lemmas), postags, pos_ids, xpostags, xpos_ids,
                                  np.array(featss), heads, types, type_ids,
                                  np.array(depss), np.array(miscs))


class CoNLL03Reader(object):
    def __init__(self, file_path, word_alphabet, char_alphabet, pos_alphabet, xpos_alphabet, chunk_alphabet, ner_alphabet):
        self.__source_file = open(file_path, 'r')
        self.__word_alphabet = word_alphabet
        self.__char_alphabet = char_alphabet
        self.__pos_alphabet = pos_alphabet
        self.__chunk_alphabet = chunk_alphabet
        self.__ner_alphabet = ner_alphabet

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
        word_ids = []
        char_seqs = []
        char_id_seqs = []
        postags = []
        pos_ids = []
        chunk_tags = []
        chunk_ids = []
        ner_tags = []
        ner_ids = []

        for tokens in lines:
            if '-' in tokens[0] or '.' in tokens[0]: # conllu clitics. Attardi
                continue
            chars = []
            char_ids = []
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.__char_alphabet.get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            char_seqs.append(chars)
            char_id_seqs.append(char_ids)

            word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
            pos = tokens[2]
            chunk = tokens[3]
            ner = tokens[4]

            words.append(word)
            word_ids.append(self.__word_alphabet.get_index(word))

            postags.append(pos)
            pos_ids.append(self.__pos_alphabet.get_index(pos))

            chunk_tags.append(chunk)
            chunk_ids.append(self.__chunk_alphabet.get_index(chunk))

            ner_tags.append(ner)
            ner_ids.append(self.__ner_alphabet.get_index(ner))

        return NERInstance(Sentence(words, word_ids, char_seqs, char_id_seqs),
                           postags, pos_ids, chunk_tags, chunk_ids, ner_tags, ner_ids)
