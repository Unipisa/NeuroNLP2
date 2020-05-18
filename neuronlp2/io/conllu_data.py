__author__ = 'max'

import os.path
import numpy as np
from collections import defaultdict, OrderedDict
import torch

from neuronlp2.io.reader import CoNLLUReader
from neuronlp2.io.alphabet import Alphabet
from neuronlp2.io.logger import get_logger
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH, UNK_ID
from neuronlp2.io.common import PAD_CHAR, PAD, PAD_UPOS, PAD_XPOS, PAD_TYPE, PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD
from neuronlp2.io.common import ROOT, END, ROOT_CHAR, ROOT_UPOS, ROOT_XPOS, ROOT_TYPE, END_CHAR, END_UPOS, END_XPOS, END_TYPE

# Special vocabulary symbols - we always put them at the start.
_START_VOCAB = [PAD, ROOT, END]
NUM_SYMBOLIC_TAGS = 3

def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=100000, embedd_dict=None,
                     min_occurrence=1, normalize_digits=True):

    def expand_vocab():
        vocab_set = set(vocab_list)
        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0 or line.startswith('#'): # conllu format. Attardi:
                        continue

                    tokens = line.split('\t')
                    if '-' in tokens[0] or '.' in tokens[0]: # conllu. Attardi
                        continue
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                    upos = tokens[3]
                    xpos = tokens[4]
                    type = tokens[7]

                    upos_alphabet.add(upos)
                    xpos_alphabet.add(xpos)
                    type_alphabet.add(type)

                    if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                        vocab_set.add(word)
                        vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', default_value=True, singleton=True)
    char_alphabet = Alphabet('character', default_value=True)
    upos_alphabet = Alphabet('upos')
    xpos_alphabet = Alphabet('xpos')
    type_alphabet = Alphabet('type')
    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD_CHAR)
        upos_alphabet.add(PAD_UPOS)
        xpos_alphabet.add(PAD_XPOS)
        type_alphabet.add(PAD_TYPE)

        char_alphabet.add(ROOT_CHAR)
        upos_alphabet.add(ROOT_UPOS)
        xpos_alphabet.add(ROOT_XPOS)
        type_alphabet.add(ROOT_TYPE)

        char_alphabet.add(END_CHAR)
        upos_alphabet.add(END_UPOS)
        xpos_alphabet.add(END_XPOS)
        type_alphabet.add(END_TYPE)

        vocab = defaultdict(int)
        with open(train_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0 or line.startswith('#'): # conllu. Attardi
                    continue

                tokens = line.split('\t')
                if '-' in tokens[0] or '.' in tokens[0]: # conllu. Attardi
                    continue
                for char in tokens[1]:
                    char_alphabet.add(char)

                word = DIGIT_RE.sub("0", tokens[1]) if normalize_digits else tokens[1]
                vocab[word] += 1

                upos = tokens[3]
                upos_alphabet.add(upos)

                xpos = tokens[4]
                xpos_alphabet.add(xpos)

                type = tokens[7]
                type_alphabet.add(type)

        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurrence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            assert isinstance(embedd_dict, OrderedDict)
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurrence

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurrence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        upos_alphabet.save(alphabet_directory)
        xpos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        upos_alphabet.load(alphabet_directory)
        xpos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    upos_alphabet.close()
    xpos_alphabet.close()
    type_alphabet.close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("UPOS Alphabet Size: %d" % upos_alphabet.size())
    logger.info("XPOS Alphabet Size: %d" % xpos_alphabet.size())
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())
    return word_alphabet, char_alphabet, upos_alphabet, xpos_alphabet, type_alphabet


