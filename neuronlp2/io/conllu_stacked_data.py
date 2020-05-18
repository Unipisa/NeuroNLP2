__author__ = 'max'

import math
import numpy as np
import torch
from neuronlp2.io.reader import CoNLLUReader
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH, UNK_ID
from neuronlp2.io.common import PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 140]

NUM_CHAR_PAD = 2

# def _obtain_child_index_for_left2right(heads):
#     child_ids = [[] for _ in range(len(heads))]
#     # skip the symbolic root.
#     for child in range(1, len(heads)):
#         head = heads[child]
#         child_ids[head].append(child)
#     return child_ids


# def _obtain_child_index_for_inside_out(heads):
#     child_ids = [[] for _ in range(len(heads))]
#     for head in range(len(heads)):
#         # first find left children inside-out
#         for child in reversed(range(1, head)):
#             if heads[child] == head:
#                 child_ids[head].append(child)
#         # second find right children inside-out
#         for child in range(head + 1, len(heads)):
#             if heads[child] == head:
#                 child_ids[head].append(child)
#     return child_ids


# def _obtain_child_index_for_depth(heads, reverse):
#     def calc_depth(head):
#         children = child_ids[head]
#         max_depth = 0
#         for child in children:
#             depth = calc_depth(child)
#             child_with_depth[head].append((child, depth))
#             max_depth = max(max_depth, depth + 1)
#         child_with_depth[head] = sorted(child_with_depth[head], key=lambda x: x[1], reverse=reverse)
#         return max_depth

#     child_ids = _obtain_child_index_for_left2right(heads)
#     child_with_depth = [[] for _ in range(len(heads))]
#     calc_depth(0)
#     return [[child for child, depth in child_with_depth[head]] for head in range(len(heads))]

def _generate_stack_inputs(heads, types):

    length = len(heads)
    stacked_heads = list(range(1, length))
    children = [0] * (length-1)
    siblings = []
    stacked_types = []
    skip_connect = []
    previous = list(range(length-1))
    next = list(range(1, length))
    prev = [0] + list(range(length-1))
    sibs = [0] * length

    for child in range(1, length): # skip root
        head = heads[child]
        siblings.append(sibs[head]) # ?
        skip_connect.append(prev[head])
        prev[head] = child
        children[child-1] = head
        sibs[head] = child      # ?
        stacked_types.append(types[child])

    return (stacked_heads, children, siblings, stacked_types, skip_connect, previous, next)


class FeatureExtractor():
    def __init__(self, word_alphabet, char_alphabet, upos_alphabet, xpos_alphabet, type_alphabet,
                 max_size=math.inf, normalize_digits=True):
        self.word_alphabet = word_alphabet
        self.char_alphabet = char_alphabet
        self.upos_alphabet = upos_alphabet
        self.xpos_alphabet = xpos_alphabet
        self.type_alphabet = type_alphabet
        self.max_size = max_size
        self.normalize_digits = normalize_digits


    def encode(self, sent):
        """
        Convert :param sent: to indices.
        """
        word_ids = [self.word_alphabet.get_index(w) for w in sent.words]
        char_id_seqs = [[self.char_alphabet.get_index(char) for char in chars[:MAX_CHAR_LENGTH]] for chars in sent.char_seqs]
        upos_ids = [self.upos_alphabet.get_index(w) for w in sent.upostags]
        xpos_ids = [self.xpos_alphabet.get_index(w) for w in sent.xpostags]
        type_ids = [self.type_alphabet.get_index(w) for w in sent.types]

        return word_ids, char_id_seqs, upos_ids, xpos_ids, sent.heads, type_ids


    # def read_data_tensor(self, source_path)
    #     """
    #     Read data from file :param source_path: in CoNLLU format.
    #     :return: a dict of tensors
    #     """
    #     data = []
    #     max_length = 0
    #     max_char_length = 0
    #     print('Reading data from %s' % source_path)
    #     counter = 0
    #     reader = CoNLLUReader(source_path, normalize_digits=normalize_digits,
    #                           symbolic_root=True, symbolic_end=False)
    #     sent = reader.getNext()
    #     while sent is not None and (not max_size or counter < max_size):
    #         counter += 1
    #         if counter % 10000 == 0:
    #             print("reading data: %d" % counter)

    #         encoded_sent = self.encode(sent)
    #         wids, cid_seqs, pids, xids, hids, tids = encoded_sent
    #         stacked_features = _generate_stack_inputs(entinst.heads, tids)
    #         data.append((encoded_sent, stacked_features))
    #         max_len = max([len(char_seq) for char_seq in sent.char_seqs])
    #         if max_char_length < max_len:
    #             max_char_length = max_len
    #         if max_length < inst.length():
    #             max_length = inst.length()
    #         sent = reader.getNext()
    #     reader.close()
    #     print("Total number of data: %d" % counter)

    #     data_size = len(data)
    #     char_length = min(MAX_CHAR_LENGTH, max_char_length)
    #     wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    #     cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
    #     pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    #     hid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    #     tid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    #     masks_e = np.zeros([data_size, max_length], dtype=np.float32)
    #     single = np.zeros([data_size, max_length], dtype=np.int64)
    #     lengths = np.empty(data_size, dtype=np.int64)

    #     stack_hid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    #     chid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    #     ssid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    #     stack_tid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    #     skip_connect_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)

    #     masks_d = np.zeros([data_size, 2 * max_length - 1], dtype=np.float32)

    #     for i, (encoded_sent, stacked_inputs) in enumerate(data):
    #         wids, cid_seqs, pids, xids, hids, tids = encoded_sent
    #         stack_hids, chids, ssids, stack_tids, skip_ids = stacked_inputs
    #         inst_size = len(wids)
    #         lengths[i] = inst_size
    #         # word ids
    #         wid_inputs[i, :inst_size] = wids
    #         wid_inputs[i, inst_size:] = PAD_ID_WORD
    #         for c, cids in enumerate(cid_seqs):
    #             cid_inputs[i, c, :len(cids)] = cids
    #             cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
    #         cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
    #         # upos ids
    #         pid_inputs[i, :inst_size] = pids
    #         pid_inputs[i, inst_size:] = PAD_ID_TAG
    #         # xpos ids
    #         xid_inputs[i, :inst_size] = xids
    #         xid_inputs[i, inst_size:] = PAD_ID_TAG
    #         # type ids
    #         tid_inputs[i, :inst_size] = tids
    #         tid_inputs[i, inst_size:] = PAD_ID_TAG
    #         # heads
    #         hid_inputs[i, :inst_size] = hids
    #         hid_inputs[i, inst_size:] = PAD_ID_TAG
    #         # masks_e
    #         masks_e[i, :inst_size] = 1.0
    #         for j, wid in enumerate(wids):
    #             if word_alphabet.is_singleton(wid):
    #                 single[i, j] = 1

    #         inst_size_decoder = 2 * inst_size - 1
    #         # stacked heads
    #         stack_hid_inputs[i, :inst_size_decoder] = stack_hids
    #         stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
    #         # children
    #         chid_inputs[i, :inst_size_decoder] = chids
    #         chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
    #         # siblings
    #         ssid_inputs[i, :inst_size_decoder] = ssids
    #         ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
    #         # stacked types
    #         stack_tid_inputs[i, :inst_size_decoder] = stack_tids
    #         stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
    #         # skip connects
    #         skip_connect_inputs[i, :inst_size_decoder] = skip_ids
    #         skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
    #         # masks_d
    #         masks_d[i, :inst_size_decoder] = 1.0

    #     words = torch.from_numpy(wid_inputs)
    #     chars = torch.from_numpy(cid_inputs)
    #     upos = torch.from_numpy(pid_inputs)
    #     xpos = torch.from_numpy(xid_inputs)
    #     heads = torch.from_numpy(hid_inputs)
    #     types = torch.from_numpy(tid_inputs)
    #     masks_e = torch.from_numpy(masks_e)
    #     single = torch.from_numpy(single)
    #     lengths = torch.from_numpy(lengths)

    #     stacked_heads = torch.from_numpy(stack_hid_inputs)
    #     children = torch.from_numpy(chid_inputs)
    #     siblings = torch.from_numpy(ssid_inputs)
    #     stacked_types = torch.from_numpy(stack_tid_inputs)
    #     skip_connect = torch.from_numpy(skip_connect_inputs)
    #     masks_d = torch.from_numpy(masks_d)

    #     tensor_dict = {'WORD': words, 'CHAR': chars, 'UPOS': pos, 'XPOS': xpos,
    #                    'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
    #                    'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
    #                    'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d}
    #     return tensor_dict, data_size


    def read_encoded_buckets(self, source_path, bucket_lengths=_buckets):
        """
        Read data from file :param source_path: in CoNLLU format.
        Convert input to indices from alphabets.
        :return: encoded sentences in buckets by length.
        """

        buckets = [[] for _ in bucket_lengths]
        max_char_lengths = [0 for _ in bucket_lengths]
        print('Reading data from %s' % source_path)
        counter = 0
        reader = CoNLLUReader(source_path, normalize_digits=self.normalize_digits,
                              symbolic_root=True, symbolic_end=False)
        sent = reader.getNext()
        while sent is not None and counter < self.max_size:
            counter += 1
            if counter % 10000 == 0:
                print("reading data: %d" % counter)

            sent_len = len(sent)
            # Warning: sentences longer than bucket_lengths[-1] are discarded
            for bucket_id, bucket_length in enumerate(bucket_lengths):
                if sent_len < bucket_length:
                    sent_inputs = self.encode(sent)
                    type_ids = sent_inputs[-1] # last
                    stacked_inputs = _generate_stack_inputs(sent.heads, type_ids)
                    buckets[bucket_id].append((sent_inputs, stacked_inputs))
                    max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                    if max_char_lengths[bucket_id] < max_len:
                        max_char_lengths[bucket_id] = max_len
                    break

            sent = reader.getNext()
        reader.close()
        print("Total number of data: %d" % counter)
        return buckets, max_char_lengths


    def read_bucketed_data(self, source_path, bucket_lengths=_buckets, use_gpu=False):
        """
        Read data from file :param source_path: in CoNLLU format.
        :param bucket_lengths: controls the length of sents in each bucket.
        :return: tensors of pairs of encoded (sentence_features, stacked_inputs) grouped in
        buckets by length.
        """

        buckets, max_char_length = self.read_encoded_buckets(source_path, bucket_lengths)
        bucket_sizes = [len(bucket) for bucket in buckets]

        data_buckets = []

        for bucket_id, bucket_length in enumerate(bucket_lengths):
            bucket_size = len(buckets[bucket_id])
            if bucket_size == 0:
                data_buckets.append((1, 1))
                continue

            char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
            # Only Tensors of floating point dtype can require gradients
            wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            # int64 or error: Expected tensor for argument #1 'indices' to have scalar type Long
            cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
            pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            xid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
            hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int32)
            tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int32)

            masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
            single = np.zeros([bucket_size, bucket_length], dtype=np.int32)
            lengths_e = np.empty(bucket_size, dtype=np.int64)

            stack_hid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
            chid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
            ssid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int32)
            stack_tid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
            skip_connect_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int32)
            previous_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
            next_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)

            masks_d = np.zeros([bucket_size, bucket_length - 1], dtype=np.float32)
            lengths_d = np.empty(bucket_size, dtype=np.int32)

            for i, encoded_sent in enumerate(buckets[bucket_id]):
                sent_inputs, stacked_inputs = encoded_sent
                word_ids, char_id_seqs, upos_ids, xpos_ids, heads, type_ids = sent_inputs
                stack_hids, chids, ssids, stack_tids, skip_ids, previous_ids, next_ids = stacked_inputs
                sent_len = len(word_ids)
                lengths_e[i] = sent_len
                # word ids
                wid_inputs[i, :sent_len] = word_ids
                wid_inputs[i, sent_len:] = PAD_ID_WORD
                for c, cids in enumerate(char_id_seqs):
                    cid_inputs[i, c, :len(cids)] = cids
                    cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
                cid_inputs[i, sent_len:, :] = PAD_ID_CHAR
                # upos ids
                pid_inputs[i, :sent_len] = upos_ids
                pid_inputs[i, sent_len:] = PAD_ID_TAG
                # xpos ids
                xid_inputs[i, :sent_len] = xpos_ids
                xid_inputs[i, sent_len:] = PAD_ID_TAG
                # type ids
                tid_inputs[i, :sent_len] = type_ids
                tid_inputs[i, sent_len:] = PAD_ID_TAG
                # heads
                hid_inputs[i, :sent_len] = heads
                hid_inputs[i, sent_len:] = PAD_ID_TAG
                # masks_e (mask for ending pads)
                masks_e[i, :sent_len] = 1.0
                for j, wid in enumerate(word_ids):
                    if self.word_alphabet.is_singleton(wid):
                        single[i, j] = 1

                decoder_sent_len = sent_len - 1        
                lengths_d[i] = decoder_sent_len
                # stacked heads
                stack_hid_inputs[i, :decoder_sent_len] = stack_hids
                stack_hid_inputs[i, decoder_sent_len:] = PAD_ID_TAG
                # children
                chid_inputs[i, :decoder_sent_len] = chids
                chid_inputs[i, decoder_sent_len:] = PAD_ID_TAG
                # siblings
                ssid_inputs[i, :decoder_sent_len] = ssids
                ssid_inputs[i, decoder_sent_len:] = PAD_ID_TAG
                # stacked types
                stack_tid_inputs[i, :decoder_sent_len] = stack_tids
                stack_tid_inputs[i, decoder_sent_len:] = PAD_ID_TAG
                # skip connects
                skip_connect_inputs[i, :decoder_sent_len] = skip_ids
                skip_connect_inputs[i, decoder_sent_len:] = PAD_ID_TAG
                # ADDED
                previous_inputs[i, :decoder_sent_len] = previous_ids
                previous_inputs[i, decoder_sent_len:] = PAD_ID_TAG
                next_inputs[i, :decoder_sent_len] = next_ids
                next_inputs[i, decoder_sent_len:] = PAD_ID_TAG
                # masks_d
                masks_d[i, :decoder_sent_len] = 1.0

            device = torch.device('cuda') if use_gpu else torch.device('cpu')

            words = torch.tensor(wid_inputs, device=device)
            chars = torch.tensor(cid_inputs, device=device)
            upos = torch.tensor(pid_inputs, device=device)
            xpos = torch.tensor(xid_inputs, device=device)
            heads = torch.tensor(hid_inputs, device=device)
            types = torch.tensor(tid_inputs, device=device)
            masks_e = torch.tensor(masks_e, device=device)
            single = torch.tensor(single, device=device)
            lengths_e = torch.tensor(lengths_e, device=device)

            stacked_heads = torch.tensor(stack_hid_inputs, device=device)
            children = torch.tensor(chid_inputs, device=device)
            siblings = torch.tensor(ssid_inputs, device=device)
            stacked_types = torch.tensor(stack_tid_inputs, device=device)
            skip_connect = torch.tensor(skip_connect_inputs, device=device)
            previous = torch.tensor(previous_inputs, device=device)
            next = torch.tensor(next_inputs, device=device)

            masks_d = torch.tensor(masks_d, device=device)
            lengths_d = torch.tensor(lengths_d, device=device)

            data_buckets.append(((words, chars, upos, xpos, heads, types,
                                 masks_e, single, lengths_e),
                                (stacked_heads, children, siblings, stacked_types,
                                 skip_connect, previous, next, masks_d, lengths_d)))

        return data_buckets, bucket_sizes


    def get_batch(self, data, batch_size, unk_replace=0.):
        """
        :param data: = (data_buckets, bucket_sizes),
            where each bucket = (data_encoder, data_decoder)
            where data_encoder = (words, chars, upos, xpos, heads, types, masks_e, single, lengths_e)
            where words.size() = [bucket_size, bucket_length]
            where chars.size() = [bucket_size, bucket_length, char_length]
            where data_decoder = (stacked_heads, children, siblings, stacked_types, skip_connect, previous, next, masks_d, lengths_d)
        :return: a random batch from :param data:
        """
        data_buckets, bucket_sizes = data
        total_size = float(sum(bucket_sizes))
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number = np.random.random_sample()
        bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
        bucket_length = _buckets[bucket_id]
        data_encoder, data_decoder = data_buckets[bucket_id]
        words, chars, upos, xpos, heads, types, masks_e, single, lengths_e = data_encoder
        stacked_heads, children, siblings, stacked_types, skip_connect, previous, next, masks_d, lengths_d = data_decoder
        bucket_size = bucket_sizes[bucket_id]
        batch_size = min(bucket_size, batch_size)
        index = torch.randperm(bucket_size).long()[:batch_size]
        if words.is_cuda:
            index = index.cuda()

        words = words[index]
        if unk_replace:
            ones = single.data.new(batch_size, bucket_length).fill_(1)
            noise = masks_e.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single[index] * noise)

        return (words, chars[index], upos[index], xpos[index],
                heads[index], types[index],
                masks_e[index], lengths_e[index]), \
               (stacked_heads[index], children[index], siblings[index], stacked_types[index], skip_connect[index], previous[index], next[index], masks_d[index], lengths_d[index])


    def batch_iterate(self, data, batch_size, unk_replace=0., shuffle=False):
        """
        Iterate through data, returning batches of size :param batch_size:
        of encoded sentences from :param data:.
        """
        data_buckets, bucket_sizes = data

        bucket_indices = np.arange(len(data_buckets))
        if shuffle:
            np.random.shuffle((bucket_indices))

        for bucket_id in bucket_indices:
            bucket_size = bucket_sizes[bucket_id]
            bucket_length = _buckets[bucket_id]
            if bucket_size == 0:
                continue
            data_encoder, data_decoder = data_buckets[bucket_id]
            words, chars, upos, xpos, heads, types, masks_e, single, lengths_e = data_encoder
            stacked_heads, children, siblings, stacked_types, skip_connect, previous, next, masks_d, lengths_d = data_decoder
            if unk_replace:
                ones = single.data.new(bucket_size, bucket_length).fill_(1)
                noise = masks_e.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long()
                words = words * (ones - single * noise)

            indices = None
            if shuffle:
                indices = torch.randperm(bucket_size).long()
                if words.is_cuda:
                    indices = indices.cuda()
            for start_idx in range(0, bucket_size, batch_size):
                if shuffle:
                    excerpt = indices[start_idx:start_idx + batch_size]
                else:
                    excerpt = slice(start_idx, start_idx + batch_size)
                yield (words[excerpt], chars[excerpt], upos[excerpt], xpos[excerpt],
                       heads[excerpt], types[excerpt],
                       masks_e[excerpt], lengths_e[excerpt]), \
                      (stacked_heads[excerpt], children[excerpt], siblings[excerpt], stacked_types[excerpt], skip_connect[excerpt], previous[excerpt], next[excerpt], masks_d[excerpt], lengths_d[excerpt])
