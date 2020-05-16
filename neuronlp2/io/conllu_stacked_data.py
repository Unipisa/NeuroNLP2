__author__ = 'max'

import numpy as np
import torch
# The Variable API has been deprecated: Variables are no longer necessary to use autograd with tensors. 
#from torch.autograd import Variable # LeftToRightParser
from neuronlp2.io.reader import CoNLLUReader
from neuronlp2.io.conllu_data import _buckets, NUM_SYMBOLIC_TAGS, create_alphabets
from neuronlp2.io.common import DIGIT_RE, MAX_CHAR_LENGTH, UNK_ID
from neuronlp2.io.common import PAD_ID_CHAR, PAD_ID_TAG, PAD_ID_WORD

NUM_CHAR_PAD = 2

def _obtain_child_index_for_left2right(heads):
    child_ids = [[] for _ in range(len(heads))]
    # skip the symbolic root.
    for child in range(1, len(heads)):
        head = heads[child]
        child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_inside_out(heads):
    child_ids = [[] for _ in range(len(heads))]
    for head in range(len(heads)):
        # first find left children inside-out
        for child in reversed(range(1, head)):
            if heads[child] == head:
                child_ids[head].append(child)
        # second find right children inside-out
        for child in range(head + 1, len(heads)):
            if heads[child] == head:
                child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_depth(heads, reverse):
    def calc_depth(head):
        children = child_ids[head]
        max_depth = 0
        for child in children:
            depth = calc_depth(child)
            child_with_depth[head].append((child, depth))
            max_depth = max(max_depth, depth + 1)
        child_with_depth[head] = sorted(child_with_depth[head], key=lambda x: x[1], reverse=reverse)
        return max_depth

    child_ids = _obtain_child_index_for_left2right(heads)
    child_with_depth = [[] for _ in range(len(heads))]
    calc_depth(0)
    return [[child for child, depth in child_with_depth[head]] for head in range(len(heads))]


# def _generate_stack_inputs(heads, types, prior_order):
#     if prior_order == 'deep_first':
#         child_ids = _obtain_child_index_for_depth(heads, True)
#     elif prior_order == 'shallow_first':
#         child_ids = _obtain_child_index_for_depth(heads, False)
#     elif prior_order == 'left2right':
#         child_ids = _obtain_child_index_for_left2right(heads)
#     elif prior_order == 'inside_out':
#         child_ids = _obtain_child_index_for_inside_out(heads)
#     else:
#         raise ValueError('Unknown prior order: %s' % prior_order)

#     stacked_heads = []
#     children = []
#     siblings = []
#     stacked_types = []
#     skip_connect = []
#     prev = [0 for _ in range(len(heads))]
#     sibs = [0 for _ in range(len(heads))]
#     stack = [0]
#     position = 1
#     while len(stack) > 0:
#         head = stack[-1]
#         stacked_heads.append(head)
#         siblings.append(sibs[head])
#         child_id = child_ids[head]
#         skip_connect.append(prev[head])
#         prev[head] = position
#         if len(child_id) == 0:
#             children.append(head)
#             sibs[head] = 0
#             stacked_types.append(PAD_ID_TAG)
#             stack.pop()
#         else:
#             child = child_id.pop(0)
#             children.append(child)
#             sibs[head] = child
#             stack.append(child)
#             stacked_types.append(types[child])
#         position += 1

#     return stacked_heads, children, siblings, stacked_types, skip_connect


def read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, xpos_alphabet, type_alphabet,
              max_size=None, normalize_digits=True, prior_order='inside_out'):
    """
    Read data from file :param source_path: in CoNLLU format.
    """
    data = []
    max_length = 0
    max_char_length = 0
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLUReader(source_path, word_alphabet, char_alphabet, pos_alphabet, xpos_alphabet, type_alphabet)
    sent = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    while sent is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
        data.append([sent, stacked_heads, children, siblings, stacked_types, skip_connect])
        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if max_char_length < max_len:
            max_char_length = max_len
        if max_length < inst.length():
            max_length = inst.length()
        sent = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    reader.close()
    print("Total number of data: %d" % counter)

    data_size = len(data)
    char_length = min(MAX_CHAR_LENGTH, max_char_length)
    wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    cid_inputs = np.empty([data_size, max_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    hid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    tid_inputs = np.empty([data_size, max_length], dtype=np.int64)

    masks_e = np.zeros([data_size, max_length], dtype=np.float32)
    single = np.zeros([data_size, max_length], dtype=np.int64)
    lengths = np.empty(data_size, dtype=np.int64)

    stack_hid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    chid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    ssid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    stack_tid_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)
    skip_connect_inputs = np.empty([data_size, 2 * max_length - 1], dtype=np.int64)

    masks_d = np.zeros([data_size, 2 * max_length - 1], dtype=np.float32)

    for i, inst in enumerate(data):
        wids, cid_seqs, pids, xids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst
        inst_size = len(wids)
        lengths[i] = inst_size
        # word ids
        wid_inputs[i, :inst_size] = wids
        wid_inputs[i, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[i, c, :len(cids)] = cids
            cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[i, :inst_size] = pids
        pid_inputs[i, inst_size:] = PAD_ID_TAG
        # xpos ids
        xid_inputs[i, :inst_size] = xids
        xid_inputs[i, inst_size:] = PAD_ID_TAG
        # type ids
        tid_inputs[i, :inst_size] = tids
        tid_inputs[i, inst_size:] = PAD_ID_TAG
        # heads
        hid_inputs[i, :inst_size] = hids
        hid_inputs[i, inst_size:] = PAD_ID_TAG
        # masks_e
        masks_e[i, :inst_size] = 1.0
        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[i, j] = 1

        inst_size_decoder = 2 * inst_size - 1
        # stacked heads
        stack_hid_inputs[i, :inst_size_decoder] = stack_hids
        stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # children
        chid_inputs[i, :inst_size_decoder] = chids
        chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # siblings
        ssid_inputs[i, :inst_size_decoder] = ssids
        ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # stacked types
        stack_tid_inputs[i, :inst_size_decoder] = stack_tids
        stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # skip connects
        skip_connect_inputs[i, :inst_size_decoder] = skip_ids
        skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
        # masks_d
        masks_d[i, :inst_size_decoder] = 1.0

    words = torch.from_numpy(wid_inputs)
    chars = torch.from_numpy(cid_inputs)
    pos = torch.from_numpy(pid_inputs)
    xpos = torch.from_numpy(xid_inputs)
    heads = torch.from_numpy(hid_inputs)
    types = torch.from_numpy(tid_inputs)
    masks_e = torch.from_numpy(masks_e)
    single = torch.from_numpy(single)
    lengths = torch.from_numpy(lengths)

    stacked_heads = torch.from_numpy(stack_hid_inputs)
    children = torch.from_numpy(chid_inputs)
    siblings = torch.from_numpy(ssid_inputs)
    stacked_types = torch.from_numpy(stack_tid_inputs)
    skip_connect = torch.from_numpy(skip_connect_inputs)
    masks_d = torch.from_numpy(masks_d)

    data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'XPOS': xpos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
                   'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
                   'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d}
    return data_tensor, data_size


# def read_bucketed_data(source_path, word_alphabet, char_alphabet, pos_alphabet, xpos_alphabet, type_alphabet,
#                        max_size=None, normalize_digits=True, prior_order='inside_out'):
#     data = [[] for _ in _buckets]
#     max_char_length = [0 for _ in _buckets]
#     print('Reading data from %s' % source_path)
#     counter = 0
#     reader = CoNLLUReader(source_path, word_alphabet, char_alphabet, pos_alphabet, spos_alphabet, type_alphabet)
#     inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
#     while inst is not None and (not max_size or counter < max_size):
#         counter += 1
#         if counter % 10000 == 0:
#             print("reading data: %d" % counter)

#         inst_size = inst.length()
#         sent = inst.sentence
#         for bucket_id, bucket_size in enumerate(_buckets):
#             if inst_size < bucket_size:
#                 stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
#                 data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.xpos_ids, inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect])
#                 max_len = max([len(char_seq) for char_seq in sent.char_seqs])
#                 if max_char_length[bucket_id] < max_len:
#                     max_char_length[bucket_id] = max_len
#                 break

#         inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
#     reader.close()
#     print("Total number of data: %d" % counter)

#     bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
#     data_tensors = []
#     for bucket_id in range(len(_buckets)):
#         bucket_size = bucket_sizes[bucket_id]
#         if bucket_size == 0:
#             data_tensors.append((1, 1))
#             continue

#         bucket_length = _buckets[bucket_id]
#         char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id])
#         wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
#         cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
#         pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
#         xid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
#         hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
#         tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

#         masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
#         single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
#         lengths = np.empty(bucket_size, dtype=np.int64)

#         stack_hid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
#         chid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
#         ssid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
#         stack_tid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
#         skip_connect_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)

#         masks_d = np.zeros([bucket_size, 2 * bucket_length - 1], dtype=np.float32)

#         for i, inst in enumerate(data[bucket_id]):
#             wids, cid_seqs, pids, xids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst
#             inst_size = len(wids)
#             lengths[i] = inst_size
#             # word ids
#             wid_inputs[i, :inst_size] = wids
#             wid_inputs[i, inst_size:] = PAD_ID_WORD
#             for c, cids in enumerate(cid_seqs):
#                 cid_inputs[i, c, :len(cids)] = cids
#                 cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
#             cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
#             # pos ids
#             pid_inputs[i, :inst_size] = pids
#             pid_inputs[i, inst_size:] = PAD_ID_TAG
#             # xpos ids
#             xid_inputs[i, :inst_size] = xids
#             xid_inputs[i, inst_size:] = PAD_ID_TAG
#             # type ids
#             tid_inputs[i, :inst_size] = tids
#             tid_inputs[i, inst_size:] = PAD_ID_TAG
#             # heads
#             hid_inputs[i, :inst_size] = hids
#             hid_inputs[i, inst_size:] = PAD_ID_TAG
#             # masks_e
#             masks_e[i, :inst_size] = 1.0
#             for j, wid in enumerate(wids):
#                 if word_alphabet.is_singleton(wid):
#                     single[i, j] = 1

#             inst_size_decoder = 2 * inst_size - 1
#             # stacked heads
#             stack_hid_inputs[i, :inst_size_decoder] = stack_hids
#             stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
#             # children
#             chid_inputs[i, :inst_size_decoder] = chids
#             chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
#             # siblings
#             ssid_inputs[i, :inst_size_decoder] = ssids
#             ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
#             # stacked types
#             stack_tid_inputs[i, :inst_size_decoder] = stack_tids
#             stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG
#             # skip connects
#             skip_connect_inputs[i, :inst_size_decoder] = skip_ids
#             skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG
#             # masks_d
#             masks_d[i, :inst_size_decoder] = 1.0

#         words = torch.from_numpy(wid_inputs)
#         chars = torch.from_numpy(cid_inputs)
#         pos = torch.from_numpy(pid_inputs)
#         xpos = torch.from_numpy(xid_inputs)
#         heads = torch.from_numpy(hid_inputs)
#         types = torch.from_numpy(tid_inputs)
#         masks_e = torch.from_numpy(masks_e)
#         single = torch.from_numpy(single)
#         lengths = torch.from_numpy(lengths)

#         stacked_heads = torch.from_numpy(stack_hid_inputs)
#         children = torch.from_numpy(chid_inputs)
#         siblings = torch.from_numpy(ssid_inputs)
#         stacked_types = torch.from_numpy(stack_tid_inputs)
#         skip_connect = torch.from_numpy(skip_connect_inputs)
#         masks_d = torch.from_numpy(masks_d)

#         data_tensor = {'WORD': words, 'CHAR': chars, 'POS': pos, 'XPOS': xpos, 'HEAD': heads, 'TYPE': types, 'MASK_ENC': masks_e,
#                        'SINGLE': single, 'LENGTH': lengths, 'STACK_HEAD': stacked_heads, 'CHILD': children,
#                        'SIBLING': siblings, 'STACK_TYPE': stacked_types, 'SKIP_CONNECT': skip_connect, 'MASK_DEC': masks_d}
#         data_tensors.append(data_tensor)

#     return data_tensors, bucket_sizes


# ----------------------------------------------------------------------
# from LeftToRightParse

def _generate_stack_inputs(heads, types, prior_order):
    #child_ids = _obtain_child_index_for_left2right(heads)

    debug = False
    
    stacked_heads = []
    children = [0 for _ in range(len(heads)-1)]
    siblings = []
    previous = []
    next = []
    stacked_types = []
    skip_connect = []
    prev = [0] * len(heads)
    sibs = [0] * len(heads)
    newheads = [-1] * len(heads)
    newheads[0] = 0
    #stack = [0]
    #stack = [1]
    position = 1                # same as child?

    for child in range(1, len(heads)):
        stacked_heads.append(child)
        if child == len(heads)-1:
                next.append(0)
        else:
                next.append(child+1)
        previous.append(child-1)
        head = heads[child]
        newheads[child] = head
        siblings.append(sibs[head])
        skip_connect.append(prev[head])
        prev[head] = position
        children[child-1] = head
        sibs[head] = child
        stacked_types.append(types[child])
        position += 1
        if debug: 
            print('stacked_heads', stacked_heads)
            print('stacked_types', stacked_types)
            print('siblings', siblings)
            print('sibs', sibs)
            print('children', children)
            print('prev', prev)
            print('heads', heads)
            print('newheads', newheads)
            print('next', next)
            print('previous', previous)

    if debug: exit(0)
    return (stacked_heads, children, siblings, stacked_types, skip_connect, previous, next)


def read_stacked_data(source_path, word_alphabet, char_alphabet, pos_alphabet,
                      xpos_alphabet, type_alphabet, max_size=None,
                      normalize_digits=True, prior_order='deep_first'):
    """
    Read data from file :param source_path: in CoNLLU format.
    Convert input to indices from alphabets.
    :return: sentences in buckets by length.
    """

    buckets = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLUReader(source_path, word_alphabet, char_alphabet, pos_alphabet, xpos_alphabet, type_alphabet)
    sent = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    while sent is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent_len = len(sent)
        for bucket_id, bucket_size in enumerate(_buckets):
            if sent_len < bucket_size:
                stacked_inputs = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
                buckets[bucket_id].append((sent, stacked_inputs))
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        sent = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    reader.close()
    print("Total number of data: %d" % counter)
    return buckets, max_char_length


def read_stacked_data_tensors(source_path, word_alphabet, char_alphabet, pos_alphabet,
                              xpos_alphabet, type_alphabet,
                              max_size=None, normalize_digits=True,
                              prior_order='deep_first', use_gpu=False, volatile=False):
    """
    Read data from file :param source_path: in CoNLLU format.
    :return: tensors of pairs of encoded (sentence_features, stacked_inputs) grouped in
    buckets by length.
    """

    buckets, max_char_length = read_stacked_data(source_path, word_alphabet, char_alphabet,
                                                 pos_alphabet, xpos_alphabet, type_alphabet,
                                                 max_size=max_size,
                                                 normalize_digits=normalize_digits,
                                                 prior_order=prior_order)
    bucket_sizes = [len(buckets[b]) for b in range(len(_buckets))]

    data_tensor = []

    for bucket_id, bucket_length in enumerate(_buckets):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensor.append((1, 1))
            continue

        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        xid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths_e = np.empty(bucket_size, dtype=np.int64)

        """
        stack_hid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([bucket_size, 2 * bucket_length - 1], dtype=np.int64)

        masks_d = np.zeros([bucket_size, 2 * bucket_length - 1], dtype=np.float32)
        """
        stack_hid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        previous_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        next_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)        

        masks_d = np.zeros([bucket_size, bucket_length - 1], dtype=np.float32)
        lengths_d = np.empty(bucket_size, dtype=np.int64)

        for i, sent_stack in enumerate(buckets[bucket_id]):
            sent, stacked_inputs = sent_stack
            stack_hids, chids, ssids, stack_tids, skip_ids, previous_ids, next_ids = stacked_inputs
            sent_len = len(wids)
            lengths_e[i] = sent_len
            # word ids
            wid_inputs[i, :sent_len] = sent.word_ids
            wid_inputs[i, sent_len:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = sent.char_ids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, sent_len:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :sent_len] = sent.pos_ids
            pid_inputs[i, sent_len:] = PAD_ID_TAG
            # xpos ids
            xid_inputs[i, :sent_len] = sent.xpos_ids
            xid_inputs[i, sent_len:] = PAD_ID_TAG
            # type ids
            tid_inputs[i, :sent_len] = sent.type_ids
            tid_inputs[i, sent_len:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :sent_len] = sent.heads
            hid_inputs[i, sent_len:] = PAD_ID_TAG
            # masks_e (mask for ending pads)
            masks_e[i, :sent_len] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            #decoder_sent_len = 2 * sent_len - 1
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

        words = torch.as_tensor(wid_inputs, device=device)
        chars = torch.as_tensor(cid_inputs, device=device)
        pos = torch.as_tensor(pid_inputs, device=device)
        xpos = torch.as_tensor(xid_inputs, device=device)
        heads = torch.as_tensor(hid_inputs, device=device)
        types = torch.as_tensor(tid_inputs, device=device)
        masks_e = torch.as_tensor(masks_e, device=device)
        single = torch.as_tensor(single, device=device)
        lengths_e = torch.as_tensor(lengths_e, device=device)

        stacked_heads = torch.as_tensor(stack_hid_inputs, device=device)
        children = torch.as_tensor(chid_inputs, device=device)
        siblings = torch.as_tensor(ssid_inputs, device=device)
        stacked_types = torch.as_tensor(stack_tid_inputs, device=device)
        skip_connect = torch.as_tensor(skip_connect_inputs, device=device)
        previous = torch.as_tensor(previous_inputs, device=device)
        next = torch.as_tensor(next_inputs, device=device)

        masks_d = torch.as_tensor(masks_d, device=device)
        lengths_d = torch.as_tensor(lengths_d, device=device)

        data_tensor.append(((words, chars, lemmas, pos, xpos, heads, types, depss, miscs,
                               masks_e, single, lengths_e),
                              (stacked_heads, children, siblings, stacked_types, skip_connect, previous, next, masks_d, lengths_d)))

    return data_tensor, bucket_sizes


def get_batch_stacked_tensor(data, batch_size, unk_replace=0.):
    data_tensor, bucket_sizes = data
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

    data_encoder, data_decoder = data_tensor[bucket_id]
    words, chars, lemmas, pos, xpos, heads, types, depss, miscs, masks_e, single, lengths_e = data_encoder
    stacked_heads, children, siblings, stacked_types, skip_connect, previous, next, masks_d, lengths_d = data_decoder
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    idx = torch.randperm(bucket_size).long()[:batch_size]
    if words.is_cuda:
        index = idx.cuda()

    print('idx:', idx, words, words[index], lemmas, len(idx), len(words), len(lemmas))  # DEBUG
    words = words[index]
    if unk_replace:
        ones = single.data.new(batch_size, bucket_length).fill_(1)
        noise = masks_e.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long()
        words = words * (ones - single[index] * noise)

    return (words, chars[index], lemmas[idx], pos[index], xpos[index],
            heads[index], types[index], depss[idx], miscs[idx],
            masks_e[index], lengths_e[index]), \
           (stacked_heads[index], children[index], siblings[index], stacked_types[index], skip_connect[index], previous[index], next[index], masks_d[index], lengths_d[index])


def iterate_batch_stacked_tensor(data, batch_size, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue
        data_encoder, data_decoder = data_tensor[bucket_id]
        words, lemmas, chars, pos, xpos, heads, types, depss, miscs, masks_e, single, lengths_e = data_encoder
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
            yield (words[excerpt], chars[excerpt], lemmas[excerpt], pos[excerpt], xpos[excerpt],
                   heads[excerpt], types[excerpt], depss[excerpt], miscs[excerpt],
                   masks_e[excerpt], lengths_e[excerpt]), \
                  (stacked_heads[excerpt], children[excerpt], siblings[excerpt], stacked_types[excerpt], skip_connect[excerpt], previous[excerpt], next[excerpt], masks_d[excerpt], lengths_d[excerpt])
