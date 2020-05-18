__author__ = 'max'

"""
Alphabet maps objects to integer ids. It provides two way mapping from the index to the objects.
"""
import json
import os
from neuronlp2.io.logger import get_logger

class Alphabet(object):
    def __init__(self, name, default_value=False, keep_growing=True, singleton=False):
        self.__name = name

        self.entry2index = {}
        self.entries = []
        self.default_value = default_value
        self.offset = 1 if self.default_value else 0
        self.keep_growing = keep_growing
        self.singletons = set() if singleton else None

        # Index 0 is occupied by default, all else following.
        self.default_index = 0 if self.default_value else None

        self.next_index = self.offset

        self.logger = get_logger('Alphabet')

    def add(self, entry):
        if entry not in self.entry2index:
            self.entries.append(entry)
            self.entry2index[entry] = self.next_index
            self.next_index += 1

    def add_singleton(self, id):
        if self.singletons is None:
            raise RuntimeError('Alphabet %s does not have singleton.' % self.__name)
        else:
            self.singletons.add(id)

    def add_singletons(self, ids):
        if self.singletons is None:
            raise RuntimeError('Alphabet %s does not have singleton.' % self.__name)
        else:
            self.singletons.update(ids)

    def is_singleton(self, id):
        if self.singletons is None:
            raise RuntimeError('Alphabet %s does not have singleton.' % self.__name)
        else:
            return id in self.singletons

    def get_index(self, entry):
        try:
            return self.entry2index[entry]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(entry)
                return index
            elif self.default_value:
                return self.default_index
            else:
                raise KeyError("entry not found: %s" % entry)

    def get_entry(self, index):
        if self.default_value and index == self.default_index:
            # First index is occupied by the wildcard element.
            return '<_UNK>'
        else:
            try:
                return self.entries[index - self.offset]
            except IndexError:
                raise IndexError('unknown index: %d' % index)

    def size(self):
        return len(self.entries) + self.offset

    def singleton_size(self):
        return len(self.singletons)

    def items(self):
        return self.entry2index.items()

    def enumerate_items(self, start):
        if start < self.offset or start >= self.size():
            raise IndexError("Enumerate is allowed between [%d : size of the alphabet)" % self.offset)
        return zip(range(start, len(self.entries) + self.offset), self.entries[start - self.offset:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        if self.singletons is None:
            return {'entry2index': self.entry2index, 'entries': self.entries}
        else:
            return {'entry2index': self.entry2index, 'entries': self.entries,
                    'singletions': list(self.singletons)}

    def __from_json(self, data):
        self.entries = data["entries"]
        self.entry2index = data["entry2index"]
        if 'singletions' in data:
            self.singletons = set(data['singletions'])
        else:
            self.singletons = None

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name
        try:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            json.dump(self.get_content(),
                      open(os.path.join(output_directory, saving_name + ".json"), 'w'), indent=4)
        except Exception as e:
            self.logger.warn("Alphabet is not saved: %s" % repr(e))

    def load(self, input_directory, name=None):
        """
        Load model architecture and weights from the give directory. This allow we use old models even the structure
        changes.
        :param input_directory: Directory to save model and weights
        :return:
        """
        loading_name = name if name else self.__name
        self.__from_json(json.load(open(os.path.join(input_directory, loading_name + ".json"))))
        self.next_index = len(self.entries) + self.offset
        self.keep_growing = False
