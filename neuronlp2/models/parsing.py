__author__ = 'max'

from overrides import overrides
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.nn import TreeCRF, VarGRU, VarRNN, VarLSTM, VarFastLSTM
from neuronlp2.nn import BiAffine, BiLinear, CharCNN, BiAAttention
from neuronlp2.tasks import parser
from tarjan import tarjan


class PriorOrder(Enum):
    DEPTH = 0
    INSIDE_OUT = 1
    LEFT2RIGTH = 2


class DeepBiAffine(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, xpos_dim, num_xpos, rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, embedd_xpos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), pos=True, xpos=True, activation='elu'):
        super(DeepBiAffine, self).__init__()

        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        self.xpos_embed = nn.Embedding(num_xpos, xpos_dim, _weight=embedd_xpos, padding_idx=1) if xpos else None
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
        self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels

        if rnn_mode == 'RNN':
            RNN = VarRNN
        elif rnn_mode == 'LSTM':
            RNN = VarLSTM
        elif rnn_mode == 'FastLSTM':
            RNN = VarFastLSTM
        elif rnn_mode == 'GRU':
            RNN = VarGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        dim_enc = word_dim + char_dim
        if pos:
            dim_enc += pos_dim
        if xpos:
            dim_enc += xpos_dim

        self.rnn = RNN(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

        out_dim = hidden_size * 2
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.biaffine = BiAffine(arc_space, arc_space)

        self.type_h = nn.Linear(out_dim, type_space)
        self.type_c = nn.Linear(out_dim, type_space)
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Tanh()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos, embedd_xpos)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos, embedd_xpos):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)
        if embedd_xpos is None and self.xpos_embed is not None:
            nn.init.uniform_(self.xpos_embed.weight, -0.1, 0.1)

        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)
            if self.xpos_embed is not None:
                self.xpos_embed.weight[self.xpos_embed.padding_idx].fill_(0)

        nn.init.xavier_uniform_(self.arc_h.weight)
        nn.init.constant_(self.arc_h.bias, 0.)
        nn.init.xavier_uniform_(self.arc_c.weight)
        nn.init.constant_(self.arc_c.bias, 0.)

        nn.init.xavier_uniform_(self.type_h.weight)
        nn.init.constant_(self.type_h.bias, 0.)
        nn.init.xavier_uniform_(self.type_c.weight)
        nn.init.constant_(self.type_c.bias, 0.)

    def _get_rnn_output(self, input_word, input_char, input_pos, input_xpos, mask=None):
        # [batch, length, word_dim]
        word = self.word_embed(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_cnn(self.char_embed(input_char))

        # apply dropout word on input
        word = self.dropout_in(word)
        char = self.dropout_in(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        enc = torch.cat([word, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            pos = self.pos_embed(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos)
            enc = torch.cat([enc, pos], dim=2)
        if self.xpos_embed is not None:
            # [batch, length, xpos_dim]
            xpos = self.xpos_embed(input_xpos)
            # apply dropout on input
            xpos = self.dropout_in(xpos)
            enc = torch.cat([enc, xpos], dim=2)

        # output from rnn [batch, length, hidden_size]
        output, _ = self.rnn(enc, mask)

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        # output size [batch, length, arc_space]
        arc_h = self.activation(self.arc_h(output))
        arc_c = self.activation(self.arc_c(output))

        # output size [batch, length, type_space]
        type_h = self.activation(self.type_h(output))
        type_c = self.activation(self.type_c(output))

        # apply dropout on arc
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        type = torch.cat([type_h, type_c], dim=1)
        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)

        # apply dropout on type
        # [batch, length, dim] --> [batch, 2 * length, dim]
        type = self.dropout_out(type.transpose(1, 2)).transpose(1, 2)
        type_h, type_c = type.chunk(2, 1)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        return (arc_h, arc_c), (type_h, type_c)

    def forward(self, input_word, input_char, input_pos, input_xpos, mask=None):
        # output from rnn [batch, length, dim]
        arc, type = self._get_rnn_output(input_word, input_char, input_pos, input_xpos, mask=mask)
        # [batch, length_head, length_child]
        out_arc = self.biaffine(arc[0], arc[1], mask_query=mask, mask_key=mask)
        return out_arc, type

    def loss(self, input_word, input_char, input_pos, input_xpos, heads, types, mask=None):
        # out_arc shape [batch, length_head, length_child]
        out_arc, out_type  = self(input_word, input_char, input_pos, input_xpos, mask=mask)
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type

        # get vector for heads [batch, length, type_space],
        type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            out_arc = out_arc.masked_fill(minus_mask, float('-inf'))

        # loss_arc shape [batch, length_c]
        loss_arc = self.criterion(out_arc, heads)
        loss_type = self.criterion(out_type.transpose(1, 2), types)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_arc = loss_arc * mask
            loss_type = loss_type * mask

        # [batch, length - 1] -> [batch] remove the symbolic root.
        return loss_arc[:, 1:].sum(dim=1), loss_type[:, 1:].sum(dim=1)

    def _decode_types(self, out_type, heads, leading_symbolic):
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        # get vector for heads [batch, length, type_space],
        type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)
        # remove the first #leading_symbolic types.
        out_type = out_type[:, :, leading_symbolic:]
        # compute the prediction of types [batch, length]
        _, types = out_type.max(dim=2)
        return types + leading_symbolic

    def decode_local(self, input_word, input_char, input_pos, input_xpos, mask=None, leading_symbolic=0):
        # out_arc shape [batch, length_h, length_c]
        out_arc, out_type = self(input_word, input_char, input_pos, input_xpos, mask=mask)
        batch, max_len, _ = out_arc.size()
        # set diagonal elements to -inf
        diag_mask = torch.eye(max_len, device=out_arc.device, dtype=torch.uint8).unsqueeze(0)
        out_arc.masked_fill_(diag_mask, float('-inf'))
        # set invalid positions to -inf
        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            out_arc.masked_fill_(minus_mask, float('-inf'))

        # compute naive predictions.
        # predition shape = [batch, length_c]
        _, heads = out_arc.max(dim=1)

        types = self._decode_types(out_type, heads, leading_symbolic)

        return heads.cpu().numpy(), types.cpu().numpy()

    def decode(self, input_word, input_char, input_pos, input_xpos, mask=None, leading_symbolic=0):
        """
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            input_xpos: Tensor
                the xpos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and types.

        """
        # out_arc shape [batch, length_h, length_c]
        out_arc, out_type = self(input_word, input_char, input_pos, input_xpos, mask=mask)

        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        batch, max_len, type_space = type_h.size()

        type_h = type_h.unsqueeze(2).expand(batch, max_len, max_len, type_space).contiguous()
        type_c = type_c.unsqueeze(1).expand(batch, max_len, max_len, type_space).contiguous()
        # compute output for type [batch, length_h, length_c, num_labels]
        out_type = self.bilinear(type_h, type_c)

        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            out_arc.masked_fill_(minus_mask, float('-inf'))
        # loss_arc shape [batch, length_h, length_c]
        loss_arc = F.log_softmax(out_arc, dim=1)
        # loss_type shape [batch, length_h, length_c, num_labels]
        loss_type = F.log_softmax(out_type, dim=3).permute(0, 3, 1, 2)
        # [batch, num_labels, length_h, length_c]
        energy = loss_arc.unsqueeze(1) + loss_type

        # compute lengths
        length = mask.sum(dim=1).long().cpu().numpy()
        return parser.decode_MST(energy.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)


class NeuroMST(DeepBiAffine):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, xpos_dim, num_xpos, rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, embedd_xpos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), pos=True, xpos=True, activation='elu'):
        super(NeuroMST, self).__init__(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, xpos_dim, num_xpos, rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                                       embedd_word=embedd_word, embedd_char=embedd_char, embedd_pos=embedd_pos, embedd_xpos=embedd_xpos, p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=pos, xpos=xpos, activation=activation)
        self.biaffine = None
        self.treecrf = TreeCRF(arc_space)

    def forward(self, input_word, input_char, input_pos, input_xpos, mask=None):
        # output from rnn [batch, length, dim]
        arc, type = self._get_rnn_output(input_word, input_char, input_pos, input_xpos, mask=mask)
        # [batch, length_head, length_child]
        out_arc = self.treecrf(arc[0], arc[1], mask=mask)
        return out_arc, type

    @overrides
    def loss(self, input_word, input_char, input_pos, input_xpos, heads, types, mask=None):
        # output from rnn [batch, length, dim]
        arc, out_type = self._get_rnn_output(input_word, input_char, input_pos, input_xpos, mask=mask)
        # [batch]
        loss_arc = self.treecrf.loss(arc[0], arc[1], heads, mask=mask)
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type

        # get vector for heads [batch, length, type_space],
        type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)
        loss_type = self.criterion(out_type.transpose(1, 2), types)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_type = loss_type * mask

        return loss_arc, loss_type[:, 1:].sum(dim=1)

    @overrides
    def decode(self, input_word, input_char, input_pos, input_xpos, mask=None, leading_symbolic=0):
        """
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            input_xpos: Tensor
                the xpos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and types.

        """
        # out_arc shape [batch, length_h, length_c]
        energy, out_type = self(input_word, input_char, input_pos, input_xpos, mask=mask)
        # compute lengths
        length = mask.sum(dim=1).long()
        heads, _ = parser.decode_MST(energy.cpu().numpy(), length.cpu().numpy(), leading_symbolic=leading_symbolic, labeled=False)
        types = self._decode_types(out_type, torch.from_numpy(heads).type_as(length), leading_symbolic)
        return heads, types.cpu().numpy()


class StackPtrNet(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, xpos_dim, num_xpos, rnn_mode, hidden_size,
                 encoder_layers, decoder_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, embedd_xpos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33),
                 pos=True, xpos=True, prior_order='inside_out', grandPar=False, sibling=False, activation='elu'):

        super(StackPtrNet, self).__init__()
        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        self.xpos_embed = nn.Embedding(num_xpos, xpos_dim, _weight=embedd_xpos, padding_idx=1) if xpos else None
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
        self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels

        if prior_order in ['deep_first', 'shallow_first']:
            self.prior_order = PriorOrder.DEPTH
        elif prior_order == 'inside_out':
            self.prior_order = PriorOrder.INSIDE_OUT
        elif prior_order == 'left2right':
            self.prior_order = PriorOrder.LEFT2RIGTH
        else:
            raise ValueError('Unknown prior order: %s' % prior_order)

        self.grandPar = grandPar
        self.sibling = sibling

        if rnn_mode == 'RNN':
            RNN_ENCODER = VarRNN
            RNN_DECODER = VarRNN
        elif rnn_mode == 'LSTM':
            RNN_ENCODER = VarLSTM
            RNN_DECODER = VarLSTM
        elif rnn_mode == 'FastLSTM':
            RNN_ENCODER = VarFastLSTM
            RNN_DECODER = VarFastLSTM
        elif rnn_mode == 'GRU':
            RNN_ENCODER = VarGRU
            RNN_DECODER = VarGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        dim_enc = word_dim + char_dim
        if pos:
            dim_enc += pos_dim
        if xpos:
            dim_enc += xpos_dim

        self.encoder_layers = encoder_layers
        self.encoder = RNN_ENCODER(dim_enc, hidden_size, num_layers=encoder_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

        dim_dec = hidden_size // 2
        self.src_dense = nn.Linear(2 * hidden_size, dim_dec)
        self.decoder_layers = decoder_layers
        self.decoder = RNN_DECODER(dim_dec, hidden_size, num_layers=decoder_layers, batch_first=True, bidirectional=False, dropout=p_rnn)

        self.hx_dense = nn.Linear(2 * hidden_size, hidden_size)

        self.arc_h = nn.Linear(hidden_size, arc_space) # arc dense for decoder
        self.arc_c = nn.Linear(hidden_size * 2, arc_space)  # arc dense for encoder
        self.biaffine = BiAffine(arc_space, arc_space)

        self.type_h = nn.Linear(hidden_size, type_space) # type dense for decoder
        self.type_c = nn.Linear(hidden_size * 2, type_space)  # type dense for encoder
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Tanh()

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos, embedd_xpos)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos, embedd_xpos):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)
        if embedd_xpos is None and self.xpos_embed is not None:
            nn.init.uniform_(self.xpos_embed.weight, -0.1, 0.1)

        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)
            if self.xpos_embed is not None:
                self.xpos_embed.weight[self.xpos_embed.padding_idx].fill_(0)

        nn.init.xavier_uniform_(self.arc_h.weight)
        nn.init.constant_(self.arc_h.bias, 0.)
        nn.init.xavier_uniform_(self.arc_c.weight)
        nn.init.constant_(self.arc_c.bias, 0.)

        nn.init.xavier_uniform_(self.type_h.weight)
        nn.init.constant_(self.type_h.bias, 0.)
        nn.init.xavier_uniform_(self.type_c.weight)
        nn.init.constant_(self.type_c.bias, 0.)

    def _get_encoder_output(self, input_word, input_char, input_pos, input_xpos, mask=None):
        # [batch, length, word_dim]
        word = self.word_embed(input_word)

        # [batch, length, char_length, char_dim]
        char = self.char_cnn(self.char_embed(input_char))

        # apply dropout word on input
        word = self.dropout_in(word)
        char = self.dropout_in(char)

        # concatenate word and char [batch, length, word_dim+char_filter]
        enc = torch.cat([word, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            pos = self.pos_embed(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos)
            enc = torch.cat([enc, pos], dim=2)
        if self.xpos_embed is not None:
            # [batch, length, xpos_dim]
            xpos = self.xpos_embed(input_xpos)
            # apply dropout on input
            xpos = self.dropout_in(xpos)
            enc = torch.cat([enc, xpos], dim=2)

        # output from rnn [batch, length, hidden_size]
        output, hn = self.encoder(enc, mask)
        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn

    def _get_decoder_output(self, output_enc, heads, heads_stack, siblings, hx, mask=None):
        # get vector for heads [batch, length_decoder, input_dim],
        enc_dim = output_enc.size(2)
        batch, length_dec = heads_stack.size()
        src_encoding = output_enc.gather(dim=1, index=heads_stack.unsqueeze(2).expand(batch, length_dec, enc_dim))

        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sib = siblings.gt(0).float().unsqueeze(2)
            output_enc_sibling = output_enc.gather(dim=1, index=siblings.unsqueeze(2).expand(batch, length_dec, enc_dim)) * mask_sib
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [batch, length_decoder, 1]
            gpars = heads.gather(dim=1, index=heads_stack).unsqueeze(2)
            # mask_gpar = gpars.ge(0).float()
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc.gather(dim=1, index=gpars.expand(batch, length_dec, enc_dim)) #* mask_gpar
            src_encoding = src_encoding + output_enc_gpar

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = self.activation(self.src_dense(src_encoding))
        # output from rnn [batch, length, hidden_size]
        output, hn = self.decoder(src_encoding, mask, hx=hx)
        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        raise RuntimeError('Stack Pointer Network does not implement forward')

    def _transform_decoder_init_state(self, hn):
        if isinstance(hn, tuple):
            hn, cn = hn
            _, batch, hidden_size = cn.size()
            # take the last layers
            # [batch, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            cn = torch.cat([cn[-2], cn[-1]], dim=1).unsqueeze(0)
            # take hx_dense to [1, batch, hidden_size]
            cn = self.hx_dense(cn)
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                cn = torch.cat([cn, cn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
            # hn is tanh(cn)
            hn = torch.tanh(cn)
            hn = (hn, cn)
        else:
            # take the last layers
            # [2, batch, hidden_size]
            hn = hn[-2:]
            # hn [2, batch, hidden_size]
            _, batch, hidden_size = hn.size()
            # first convert hn t0 [batch, 2, hidden_size]
            hn = hn.transpose(0, 1).contiguous()
            # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            hn = hn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [1, batch, hidden_size]
            hn = torch.tanh(self.hx_dense(hn))
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                hn = torch.cat([hn, hn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
        return hn

    def loss(self, input_word, input_char, input_pos, input_xpos, heads, stacked_heads, children, siblings, stacked_types, mask_e=None, mask_d=None):
        # output from encoder [batch, length_encoder, hidden_size]
        output_enc, hn = self._get_encoder_output(input_word, input_char, input_pos, input_xpos, mask=mask_e)

        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = self.activation(self.type_c(output_enc))

        # transform hn to [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)

        # output from decoder [batch, length_decoder, tag_space]
        output_dec, _ = self._get_decoder_output(output_enc, heads, stacked_heads, siblings, hn, mask=mask_d)

        # output size [batch, length_decoder, arc_space]
        arc_h = self.activation(self.arc_h(output_dec))
        type_h = self.activation(self.type_h(output_dec))

        batch, max_len_d, type_space = type_h.size()
        # apply dropout
        # [batch, length_decoder, dim] + [batch, length_encoder, dim] --> [batch, length_decoder + length_encoder, dim]
        arc = self.dropout_out(torch.cat([arc_h, arc_c], dim=1).transpose(1, 2)).transpose(1, 2)
        arc_h = arc[:, :max_len_d]
        arc_c = arc[:, max_len_d:]

        type = self.dropout_out(torch.cat([type_h, type_c], dim=1).transpose(1, 2)).transpose(1, 2)
        type_h = type[:, :max_len_d].contiguous()
        type_c = type[:, max_len_d:]

        # [batch, length_decoder, length_encoder]
        out_arc = self.biaffine(arc_h, arc_c, mask_query=mask_d, mask_key=mask_e)

        # get vector for heads [batch, length_decoder, type_space],
        type_c = type_c.gather(dim=1, index=children.unsqueeze(2).expand(batch, max_len_d, type_space))
        # compute output for type [batch, length_decoder, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask_e is not None:
            minus_mask_e = mask_e.eq(0).unsqueeze(1)
            minus_mask_d = mask_d.eq(0).unsqueeze(2)
            out_arc = out_arc.masked_fill(minus_mask_d * minus_mask_e, float('-inf'))

        # loss_arc shape [batch, length_decoder]
        loss_arc = self.criterion(out_arc.transpose(1, 2), children)
        loss_type = self.criterion(out_type.transpose(1, 2), stacked_types)

        if mask_d is not None:
            loss_arc = loss_arc * mask_d
            loss_type = loss_type * mask_d

        return loss_arc.sum(dim=1), loss_type.sum(dim=1)

    def decode(self, input_word, input_char, input_pos, input_xpos, mask=None, beam=1, leading_symbolic=0):
        # reset noise for decoder
        self.decoder.reset_noise(0)

        # output_enc [batch, length, model_dim]
        # arc_c [batch, length, arc_space]
        # type_c [batch, length, type_space]
        # hn [num_direction, batch, hidden_size]
        output_enc, hn = self._get_encoder_output(input_word, input_char, input_pos, input_xpos, mask=mask)
        enc_dim = output_enc.size(2)
        device = output_enc.device
        # output size [batch, length_encoder, arc_space]
        arc_c = self.activation(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = self.activation(self.type_c(output_enc))
        type_space = type_c.size(2)
        # [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)
        batch, max_len, _ = output_enc.size()

        heads = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)
        types = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)

        num_steps = 2 * max_len - 1
        stacked_heads = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64)
        siblings = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64) if self.sibling else None
        hypothesis_scores = output_enc.new_zeros((batch, 1))

        # [batch, beam, length]
        children = torch.arange(max_len, device=device, dtype=torch.int64).view(1, 1, max_len).expand(batch, beam, max_len)
        constraints = torch.zeros(batch, 1, max_len, device=device, dtype=torch.bool)
        constraints[:, :, 0] = True
        # [batch, 1]
        batch_index = torch.arange(batch, device=device, dtype=torch.int64).view(batch, 1)

        # compute lengths
        if mask is None:
            steps = torch.new_tensor([num_steps] * batch, dtype=torch.int64, device=device)
            mask_sent = torch.ones(batch, 1, max_len, dtype=torch.bool, device=device)
        else:
            steps = (mask.sum(dim=1) * 2 - 1).long()
            mask_sent = mask.unsqueeze(1).bool()

        num_hyp = 1
        mask_hyp = torch.ones(batch, 1, device=device)
        hx = hn
        for t in range(num_steps):
            # [batch, num_hyp]
            curr_heads = stacked_heads[:, :, t]
            curr_gpars = heads.gather(dim=2, index=curr_heads.unsqueeze(2)).squeeze(2)
            curr_sibs = siblings[:, :, t] if self.sibling else None
            # [batch, num_hyp, enc_dim]
            src_encoding = output_enc.gather(dim=1, index=curr_heads.unsqueeze(2).expand(batch, num_hyp, enc_dim))

            if self.sibling:
                mask_sib = curr_sibs.gt(0).float().unsqueeze(2)
                output_enc_sibling = output_enc.gather(dim=1, index=curr_sibs.unsqueeze(2).expand(batch, num_hyp, enc_dim)) * mask_sib
                src_encoding = src_encoding + output_enc_sibling

            if self.grandPar:
                output_enc_gpar = output_enc.gather(dim=1, index=curr_gpars.unsqueeze(2).expand(batch, num_hyp, enc_dim))
                src_encoding = src_encoding + output_enc_gpar

            # transform to decoder input
            # [batch, num_hyp, dec_dim]
            src_encoding = self.activation(self.src_dense(src_encoding))

            # output [batch * num_hyp, dec_dim]
            # hx [decoder_layer, batch * num_hyp, dec_dim]
            output_dec, hx = self.decoder.step(src_encoding.view(batch * num_hyp, -1), hx=hx)
            dec_dim = output_dec.size(1)
            # [batch, num_hyp, dec_dim]
            output_dec = output_dec.view(batch, num_hyp, dec_dim)

            # [batch, num_hyp, arc_space]
            arc_h = self.activation(self.arc_h(output_dec))
            # [batch, num_hyp, type_space]
            type_h = self.activation(self.type_h(output_dec))
            # [batch, num_hyp, length]
            out_arc = self.biaffine(arc_h, arc_c, mask_query=mask_hyp, mask_key=mask)
            # mask invalid position to -inf for log_softmax
            if mask is not None:
                minus_mask_enc = mask.eq(0).unsqueeze(1)
                out_arc.masked_fill_(minus_mask_enc, float('-inf'))

            # [batch]
            mask_last = steps.le(t + 1)
            mask_stop = steps.le(t)
            minus_mask_hyp = mask_hyp.eq(0).unsqueeze(2)
            # [batch, num_hyp, length]
            hyp_scores = F.log_softmax(out_arc, dim=2).masked_fill_(mask_stop.view(batch, 1, 1) + minus_mask_hyp, 0)
            # [batch, num_hyp, length]
            hypothesis_scores = hypothesis_scores.unsqueeze(2) + hyp_scores

            # [batch, num_hyp, length]
            mask_leaf = curr_heads.unsqueeze(2).eq(children[:, :num_hyp]) * mask_sent
            mask_non_leaf = (~mask_leaf) * mask_sent

            # apply constrains to select valid hyps
            # [batch, num_hyp, length]
            mask_leaf = mask_leaf * (mask_last.unsqueeze(1) + curr_heads.ne(0)).unsqueeze(2)
            mask_non_leaf = mask_non_leaf * (~constraints)

            hypothesis_scores.masked_fill_(~(mask_non_leaf + mask_leaf), float('-inf'))
            # [batch, num_hyp * length]
            hypothesis_scores, hyp_index = torch.sort(hypothesis_scores.view(batch, -1), dim=1, descending=True)

            # [batch]
            prev_num_hyp = num_hyp
            num_hyps = (mask_leaf + mask_non_leaf).long().view(batch, -1).sum(dim=1)
            num_hyp = num_hyps.max().clamp(max=beam).item()
            # [batch, hum_hyp]
            hyps = torch.arange(num_hyp, device=device, dtype=torch.int64).view(1, num_hyp)
            mask_hyp = hyps.lt(num_hyps.unsqueeze(1)).float()

            # [batch, num_hyp]
            hypothesis_scores = hypothesis_scores[:, :num_hyp]
            hyp_index = hyp_index[:, :num_hyp]
            base_index = hyp_index / max_len
            child_index = hyp_index % max_len

            # [batch, num_hyp]
            hyp_heads = curr_heads.gather(dim=1, index=base_index)
            hyp_gpars = curr_gpars.gather(dim=1, index=base_index)

            # [batch, num_hyp, length]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, max_len)
            constraints = constraints.gather(dim=1, index=base_index_expand)
            constraints.scatter_(2, child_index.unsqueeze(2), True)

            # [batch, num_hyp]
            mask_leaf = hyp_heads.eq(child_index)
            # [batch, num_hyp, length]
            heads = heads.gather(dim=1, index=base_index_expand)
            heads.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, hyp_gpars, hyp_heads).unsqueeze(2))
            types = types.gather(dim=1, index=base_index_expand)
            # [batch, num_hyp]
            org_types = types.gather(dim=2, index=child_index.unsqueeze(2)).squeeze(2)

            # [batch, num_hyp, num_steps]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, num_steps + 1)
            stacked_heads = stacked_heads.gather(dim=1, index=base_index_expand)
            stacked_heads[:, :, t + 1] = torch.where(mask_leaf, hyp_gpars, child_index)
            if self.sibling:
                siblings = siblings.gather(dim=1, index=base_index_expand)
                siblings[:, :, t + 1] = torch.where(mask_leaf, child_index, torch.zeros_like(child_index))

            # [batch, num_hyp, type_space]
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, type_space)
            child_index_expand = child_index.unsqueeze(2).expand(batch, num_hyp, type_space)
            # [batch, num_hyp, num_labels]
            out_type = self.bilinear(type_h.gather(dim=1, index=base_index_expand), type_c.gather(dim=1, index=child_index_expand))
            hyp_type_scores = F.log_softmax(out_type, dim=2)
            # compute the prediction of types [batch, num_hyp]
            hyp_type_scores, hyp_types = hyp_type_scores.max(dim=2)
            hypothesis_scores = hypothesis_scores + hyp_type_scores.masked_fill_(mask_stop.view(batch, 1), 0)
            types.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, org_types, hyp_types).unsqueeze(2))

            # hx [decoder_layer, batch * num_hyp, dec_dim]
            # hack to handle LSTM
            hx_index = (base_index + batch_index * prev_num_hyp).view(batch * num_hyp)
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, hx_index]
                cx = cx[:, hx_index]
                hx = (hx, cx)
            else:
                hx = hx[:, hx_index]

        heads = heads[:, 0].cpu().numpy()
        types = types[:, 0].cpu().numpy()
        return heads, types


#----------------------------------------------------------------------
# added from LeftToRightParser. Attardi

class NewStackPtrNet(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                 xpos_dim, num_xpos, num_filters, kernel_size,
                 rnn_mode, input_size_decoder, hidden_size,
                 encoder_layers, decoder_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, embedd_xpos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33),
                 biaffine=True,
                 pos=True, xpos=True, char=True, prior_order='inside_out', skipConnect=False, grandPar=False, sibling=False):

        super(NewStackPtrNet, self).__init__()
        self.word_embedd = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embedd = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        self.xpos_embedd = nn.Embedding(num_xpos, xpos_dim, _weight=embedd_xpos, padding_idx=1) if xpos else None
        self.char_embedd = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1) if char else None
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if char else None
        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels
        if prior_order in ['deep_first', 'shallow_first']:
            self.prior_order = PriorOrder.DEPTH
        elif prior_order == 'inside_out':
            self.prior_order = PriorOrder.INSIDE_OUT
        elif prior_order == 'left2right':
            self.prior_order = PriorOrder.LEFT2RIGTH
        else:
            raise ValueError('Unknown prior order: %s' % prior_order)
        self.pos = pos
        self.xpos = xpos
        self.char = char
        self.skipConnect = skipConnect
        self.grandPar = grandPar
        self.sibling = sibling

        if rnn_mode == 'RNN':
            RNN_ENCODER = VarRNN
            RNN_DECODER = SkipConnectRNN if skipConnect else VarRNN
        elif rnn_mode == 'LSTM':
            RNN_ENCODER = VarLSTM
            RNN_DECODER = SkipConnectLSTM if skipConnect else VarLSTM
        elif rnn_mode == 'FastLSTM':
            RNN_ENCODER = VarFastLSTM
            RNN_DECODER = SkipConnectFastLSTM if skipConnect else VarFastLSTM
        elif rnn_mode == 'GRU':
            RNN_ENCODER = VarGRU
            RNN_DECODER = SkipConnectGRU if skipConnect else VarGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        dim_enc = word_dim
        if pos:
            dim_enc += pos_dim
        if xpos:
            dim_enc += xpos_dim
        if char:
            dim_enc += num_filters

        dim_dec = input_size_decoder

        self.src_dense = nn.Linear(2 * hidden_size, dim_dec)

        self.encoder_layers = encoder_layers
        self.encoder = RNN_ENCODER(dim_enc, hidden_size, num_layers=encoder_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

        self.decoder_layers = decoder_layers
        self.decoder = RNN_DECODER(dim_dec, hidden_size, num_layers=decoder_layers, batch_first=True, bidirectional=False, dropout=p_rnn)

        self.hx_dense = nn.Linear(2 * hidden_size, hidden_size)

        self.arc_h = nn.Linear(hidden_size, arc_space) # arc dense for decoder
        self.arc_c = nn.Linear(hidden_size * 2, arc_space)  # arc dense for encoder
        self.attention = BiAAttention(arc_space, arc_space, 1, biaffine=biaffine)

        self.type_h = nn.Linear(hidden_size, type_space) # type dense for decoder
        self.type_c = nn.Linear(hidden_size * 2, type_space)  # type dense for encoder
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

    def _get_encoder_output(self, input_word, input_char, input_pos, input_xpos, mask_e=None, length_e=None, hx=None):
        # [batch, length, word_dim]
        word = self.word_embedd(input_word)
        # apply dropout on input
        word = self.dropout_in(word)

        src_encoding = word

        if self.char:
            # [batch, length, char_length, char_dim]
            char = self.char_embedd(input_char)
            char_size = char.size()
            # first transform to [batch *length, char_length, char_dim]
            # then transpose to [batch * length, char_dim, char_length]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
            # put into cnn [batch*length, char_filters, char_length]
            # then put into maxpooling [batch * length, char_filters]
            char, _ = self.conv1d(char).max(dim=2)
            # reshape to [batch, length, char_filters]
            char = torch.tanh(char).view(char_size[0], char_size[1], -1)
            # apply dropout on input
            char = self.dropout_in(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            src_encoding = torch.cat([src_encoding, char], dim=2)

        if self.pos:
            # [batch, length, pos_dim]
            pos = self.pos_embedd(input_pos)
            # apply dropout on input
            pos = self.dropout_in(pos)
            src_encoding = torch.cat([src_encoding, pos], dim=2)

        if self.xpos:
            # [batch, length, xpos_dim]
            xpos = self.xpos_embedd(input_xpos)
            # apply dropout on input
            xpos = self.dropout_in(xpos)
            src_encoding = torch.cat([src_encoding, xpos], dim=2)

        # output from rnn [batch, length, hidden_size]
        output, hn = self.encoder(src_encoding, mask_e, hx=hx)

        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn, mask_e, length_e

    def _get_decoder_output(self, output_enc, heads, heads_stack, siblings, previous, next, hx, mask_d=None, length_d=None):
        batch, _, _ = output_enc.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(output_enc.data).long()
        # get vector for heads [batch, length_decoder, input_dim],
        src_encoding = output_enc[batch_index, heads_stack.data.t()].transpose(0, 1)

        if self.sibling:#NEXT
            # [batch, length_decoder, hidden_size * 2]
            #mask_sibs = siblings.ne(0).float().unsqueeze(2)
            #output_enc_sibling = output_enc[batch_index, siblings.data.t()].transpose(0, 1) * mask_sibs
            #src_encoding = src_encoding + output_enc_sibling

            mask_next = next.ne(0).float().unsqueeze(2)
            output_enc_next = output_enc[batch_index, next.data.t()].transpose(0, 1) * mask_next
            src_encoding = src_encoding + output_enc_next

        if self.grandPar:#PREVIOUS
            # [length_decoder, batch]
            #gpars = heads[batch_index, heads_stack.data.t()].data#No tiene sentido para bottom-up
            # [batch, length_decoder, hidden_size * 2]
            #output_enc_gpar = output_enc[batch_index, gpars].transpose(0, 1)
            #src_encoding = src_encoding + output_enc_gpar
            mask_previous = previous.ne(0).float().unsqueeze(2) # Con esta mascara evitamos que tenga en cuenta que el root esta a la izquierda del primer nodo
            output_enc_previous = output_enc[batch_index, previous.data.t()].transpose(0, 1) * mask_previous
            src_encoding = src_encoding + output_enc_previous

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = F.elu(self.src_dense(src_encoding))

        # output from rnn [batch, length, hidden_size]
        output, hn = self.decoder(src_encoding, mask_d, hx=hx)

        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn, mask_d, length_d

    def _get_decoder_output_with_skip_connect(self, output_enc, heads, heads_stack, siblings, skip_connect, hx, mask_d=None, length_d=None):
        batch, _, _ = output_enc.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(output_enc.data).long()
        # get vector for heads [batch, length_decoder, input_dim],
        src_encoding = output_enc[batch_index, heads_stack.data.t()].transpose(0, 1)

        if self.sibling:
            # [batch, length_decoder, hidden_size * 2]
            mask_sibs = siblings.ne(0).float().unsqueeze(2)
            output_enc_sibling = output_enc[batch_index, siblings.data.t()].transpose(0, 1) * mask_sibs
            src_encoding = src_encoding + output_enc_sibling

        if self.grandPar:
            # [length_decoder, batch]
            gpars = heads[batch_index, heads_stack.data.t()].data
            # [batch, length_decoder, hidden_size * 2]
            output_enc_gpar = output_enc[batch_index, gpars].transpose(0, 1)
            src_encoding = src_encoding + output_enc_gpar

        # transform to decoder input
        # [batch, length_decoder, dec_dim]
        src_encoding = F.elu(self.src_dense(src_encoding))

        # output from rnn [batch, length, hidden_size]
        output, hn = self.decoder(src_encoding, skip_connect, mask_d, hx=hx)

        # apply dropout
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)

        return output, hn, mask_d, length_d

    def forward(self, input_word, input_char, input_pos, input_xpos, mask=None, length=None, hx=None):
        raise RuntimeError('Stack Pointer Network does not implement forward')

    def _transform_decoder_init_state(self, hn):
        if isinstance(hn, tuple):
            hn, cn = hn
            # take the last layers
            # [2, batch, hidden_size]
            cn = cn[-2:]
            # hn [2, batch, hidden_size]
            _, batch, hidden_size = cn.size()
            # first convert cn t0 [batch, 2, hidden_size]
            cn = cn.transpose(0, 1).contiguous()
            # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            cn = cn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [1, batch, hidden_size]
            cn = self.hx_dense(cn)
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                cn = torch.cat([cn, hn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
            # hn is tanh(cn)
            hn = F.tanh(cn)
            hn = (hn, cn)
        else:
            # take the last layers
            # [2, batch, hidden_size]
            hn = hn[-2:]
            # hn [2, batch, hidden_size]
            _, batch, hidden_size = hn.size()
            # first convert hn t0 [batch, 2, hidden_size]
            hn = hn.transpose(0, 1).contiguous()
            # then view to [batch, 1, 2 * hidden_size] --> [1, batch, 2 * hidden_size]
            hn = hn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            # take hx_dense to [1, batch, hidden_size]
            hn = F.tanh(self.hx_dense(hn))
            # [decoder_layers, batch, hidden_size]
            if self.decoder_layers > 1:
                hn = torch.cat([hn, hn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
        return hn

    def loss(self, input_word, input_char, input_pos, input_xpos, heads, stacked_heads, children, siblings, stacked_types, previous, next, label_smooth,
             skip_connect=None, mask_e=None, length_e=None, mask_d=None, length_d=None, hx=None):
        # output from encoder [batch, length_encoder, hidden_size]
        output_enc, hn, mask_e, _ = self._get_encoder_output(input_word, input_char, input_pos, input_xpos, mask_e=mask_e, length_e=length_e, hx=hx)

        # output size [batch, length_encoder, arc_space]
        arc_c = F.elu(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = F.elu(self.type_c(output_enc))

        # transform hn to [decoder_layers, batch, hidden_size]
        hn = self._transform_decoder_init_state(hn)

        # output from decoder [batch, length_decoder, tag_space]
        if self.skipConnect:
            output_dec, _, mask_d, _ = self._get_decoder_output_with_skip_connect(output_enc, heads, stacked_heads, siblings, skip_connect, hn, mask_d=mask_d, length_d=length_d)
        else:
            output_dec, _, mask_d, _ = self._get_decoder_output(output_enc, heads, stacked_heads, siblings, previous, next, hn, mask_d=mask_d, length_d=length_d)

        # output size [batch, length_decoder, arc_space]
        arc_h = F.elu(self.arc_h(output_dec))
        type_h = F.elu(self.type_h(output_dec))

        _, max_len_d, _ = arc_h.size()

        if mask_d is not None and children.size(1) != mask_d.size(1):
            stacked_heads = stacked_heads[:, :max_len_d]
            children = children[:, :max_len_d]
            stacked_types = stacked_types[:, :max_len_d]

        # apply dropout
        # [batch, length_decoder, dim] + [batch, length_encoder, dim] --> [batch, length_decoder + length_encoder, dim]
        arc = self.dropout_out(torch.cat([arc_h, arc_c], dim=1).transpose(1, 2)).transpose(1, 2)
        arc_h = arc[:, :max_len_d]
        arc_c = arc[:, max_len_d:]

        type = self.dropout_out(torch.cat([type_h, type_c], dim=1).transpose(1, 2)).transpose(1, 2)
        type_h = type[:, :max_len_d].contiguous()
        type_c = type[:, max_len_d:]

        # [batch, length_decoder, length_encoder]
        out_arc = self.attention(arc_h, arc_c, mask_d=mask_d, mask_e=mask_e).squeeze(dim=1)

        batch, max_len_e, _ = arc_c.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(arc_c.data).long()

        # get vector for heads [batch, length_decoder, type_space],
        type_c = type_c[batch_index, children.data.t()].transpose(0, 1).contiguous() 

        # compute output for type [batch, length_decoder, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask_e is not None:
            minus_inf = -1e8
            minus_mask_d = (1 - mask_d) * minus_inf
            minus_mask_e = (1 - mask_e) * minus_inf
            out_arc = out_arc + minus_mask_d.unsqueeze(2) + minus_mask_e.unsqueeze(1)

        # [batch, length_decoder, length_encoder]
        loss_arc = F.log_softmax(out_arc, dim=2)
        # [batch, length_decoder, num_labels]
        loss_type = F.log_softmax(out_type, dim=2)

        # compute coverage loss
        # [batch, length_decoder, length_encoder]
        coverage = torch.exp(loss_arc).cumsum(dim=1)

        # mask invalid position to 0 for sum loss
        if mask_e is not None:
            loss_arc = loss_arc * mask_d.unsqueeze(2) * mask_e.unsqueeze(1)
            coverage = coverage * mask_d.unsqueeze(2) * mask_e.unsqueeze(1)
            loss_type = loss_type * mask_d.unsqueeze(2)
           
            num = mask_d.sum()
        else:
         
            num = max_len_e

        # first create index matrix [length, batch]
        head_index = torch.arange(0, max_len_d).view(max_len_d, 1).expand(max_len_d, batch)

        head_index = head_index.type_as(out_arc.data).long()
        # [batch, length_decoder]
        if 0.0 < label_smooth < 1.0 - 1e-4:
            # label smoothing
            loss_arc1 = loss_arc[batch_index, head_index, children.data.t()].transpose(0, 1)
            loss_arc2 = loss_arc.sum(dim=2) / mask_e.sum(dim=1).unsqueeze(1)
            loss_arc = loss_arc1 * label_smooth + loss_arc2 * (1 - label_smooth)

            loss_type1 = loss_type[batch_index, head_index, stacked_types.data.t()].transpose(0, 1)
            loss_type2 = loss_type.sum(dim=2) / self.num_labels
            loss_type = loss_type1 * label_smooth + loss_type2 * (1 - label_smooth)
        else:
            loss_arc = loss_arc[batch_index, head_index, children.data.t()].transpose(0, 1)
            loss_type = loss_type[batch_index, head_index, stacked_types.data.t()].transpose(0, 1)

        loss_cov = (coverage - 2.0).clamp(min=0.)

        return -loss_arc.sum() / num,\
               -loss_type.sum() / num, \
                loss_cov.sum() / num, num


    def _decode_per_sentence(self, output_enc, arc_c, type_c, hx, length, beam, ordered, leading_symbolic):
        def hasCycles(A, head, dep):

                if head == dep: return True

                aux = set(A)
                aux.add((head,dep))
                if count_cycles(aux) != 0: 
                        return True
                return False

        def count_cycles(A):
        
                d = {}
                for a,b in A:
                    if a not in d:
                        d[a] = [b]
                    else:
                        d[a].append(b)
                   
                return sum([1 for e in tarjan(d) if len(e) > 1])


        debug = False

        # output_enc [length, hidden_size * 2]
        # arc_c [length, arc_space]
        # type_c [length, type_space]
        # hx [decoder_layers, hidden_size]
        if length is not None:
            output_enc = output_enc[:length]
            arc_c = arc_c[:length]
            type_c = type_c[:length]
        else:
            length = output_enc.size(0)

        # [decoder_layers, 1, hidden_size]
        # hack to handle LSTM
        if isinstance(hx, tuple):
            hx, cx = hx
            hx = hx.unsqueeze(1)
            cx = cx.unsqueeze(1)
            h0 = hx
            hx = (hx, cx)
        else:
            hx = hx.unsqueeze(1)
            h0 = hx

        #stacked_heads = [[0] for _ in range(beam)]
        stacked_heads = [[1] for _ in range(beam)]
        grand_parents = [[0] for _ in range(beam)] if self.grandPar else None
        if length > 2:
                siblings = [[2] for _ in range(beam)] if self.sibling else None
        else:
                siblings = [[0] for _ in range(beam)] if self.sibling else None
        skip_connects = [[h0] for _ in range(beam)] if self.skipConnect else None
        children = torch.zeros(beam,length - 1).type_as(output_enc.data).long()
        stacked_types = children.new(children.size()).zero_()
        hypothesis_scores = output_enc.data.new(beam).zero_()
        positions = [1 for _ in range(beam)]
        arcs = [set([]) for _ in range(beam)]

        # temporal tensors for each step.
        new_stacked_heads = [[] for _ in range(beam)]
        new_grand_parents = [[] for _ in range(beam)] if self.grandPar else None
        new_siblings = [[] for _ in range(beam)] if self.sibling else None
        new_skip_connects = [[] for _ in range(beam)] if self.skipConnect else None
        new_children = children.new(children.size()).zero_()
        new_stacked_types = stacked_types.new(stacked_types.size()).zero_()
        num_hyp = 1
        num_step = length - 1

        new_arcs = [set([]) for _ in range(beam)]
        new_positions = [1 for _ in range(beam)]

        for t in range(num_step):
            heads = torch.LongTensor([stacked_heads[i][-1] for i in range(num_hyp)]).type_as(children)
            gpars = torch.LongTensor([grand_parents[i][-1] for i in range(num_hyp)]).type_as(children) if self.grandPar else None
            sibs = torch.LongTensor([siblings[i][-1] for i in range(num_hyp)]).type_as(children) if self.sibling else None

            # [decoder_layers, num_hyp, hidden_size]
            hs = torch.cat([skip_connects[i].pop() for i in range(num_hyp)], dim=1) if self.skipConnect else None

            # [num_hyp, hidden_size * 2]
            src_encoding = output_enc[heads]

            if self.sibling:
                mask_sibs = sibs.ne(0).float().unsqueeze(1)
                output_enc_sibling = output_enc[sibs] * mask_sibs
                src_encoding = src_encoding + output_enc_sibling

            if self.grandPar:
                mask_gpar = gpars.ne(0).float().unsqueeze(1)
                output_enc_gpar = output_enc[gpars] * mask_gpar
                src_encoding = src_encoding + output_enc_gpar

            # transform to decoder input
            # [num_hyp, dec_dim]
            src_encoding = F.elu(self.src_dense(src_encoding))

            # output [num_hyp, hidden_size]
            # hx [decoder_layer, num_hyp, hidden_size]
            output_dec, hx = self.decoder.step(src_encoding, hx=hx, hs=hs) if self.skipConnect else self.decoder.step(src_encoding, hx=hx)

            # arc_h size [num_hyp, 1, arc_space]
            arc_h = F.elu(self.arc_h(output_dec.unsqueeze(1)))
            # type_h size [num_hyp, type_space]
            type_h = F.elu(self.type_h(output_dec))

            # [num_hyp, length_encoder]
            out_arc = self.attention(arc_h, arc_c.expand(num_hyp, *arc_c.size())).squeeze(dim=1).squeeze(dim=1)


            # [num_hyp, length_encoder]
            hyp_scores = F.log_softmax(out_arc, dim=1).data

            new_hypothesis_scores = hypothesis_scores[:num_hyp].unsqueeze(1) + hyp_scores
            # [num_hyp * length_encoder]
            new_hypothesis_scores, hyp_index = torch.sort(new_hypothesis_scores.view(-1), dim=0, descending=True)

            base_index = hyp_index / length
            child_index = hyp_index % length 

            cc = 0
            ids = []
            #new_constraints = np.zeros([beam, length], dtype=np.bool)

            for id in range(num_hyp * length):

                base_id = base_index[id]
                child_id = child_index[id]
                head = heads[base_id]
                new_hyp_score = new_hypothesis_scores[id]

                if hasCycles(arcs[base_id], child_id, head): 
                        continue

                new_arcs[cc] = set(arcs[base_id])
                new_arcs[cc].add((child_id,head))

                new_positions[cc]=positions[base_id]
                new_positions[cc]+=1
                if new_positions[cc] == length: next_position=1
                new_stacked_heads[cc] = [stacked_heads[base_id][i] for i in range(len(stacked_heads[base_id]))]
                new_stacked_heads[cc].append(new_positions[cc])

                if self.grandPar:
                        previous_position=new_positions[cc]-1
                        new_grand_parents[cc] = [grand_parents[base_id][i] for i in range(len(grand_parents[base_id]))]
                        new_grand_parents[cc].append(previous_position)
                        
                if self.sibling:
                        next_position = new_positions[cc]+1
                        if next_position == length: next_position=0
                        new_siblings[cc] = [siblings[base_id][i] for i in range(len(siblings[base_id]))]
                        new_siblings[cc].append(next_position)

                if self.skipConnect:
                        new_skip_connects[cc] = [skip_connects[base_id][i] for i in range(len(skip_connects[base_id]))]

                new_children[cc] = children[base_id]
                new_children[cc, head-1] = child_id
                hypothesis_scores[cc] = new_hyp_score
                ids.append(id)
                cc += 1

                if cc == beam:
                    break

            # [num_hyp]
            num_hyp = len(ids)
            if num_hyp == 0:
                return None
            elif num_hyp == 1:
                index = base_index.new(1).fill_(ids[0])
            else:
                index = torch.from_numpy(np.array(ids)).type_as(base_index)

            base_index = base_index[index]
            child_index = child_index[index]

            # predict types for new hypotheses
            # compute output for type [num_hyp, num_labels]
            out_type = self.bilinear(type_h[base_index], type_c[child_index])
            hyp_type_scores = F.log_softmax(out_type, dim=1).data
            # compute the prediction of types [num_hyp]
            hyp_type_scores, hyp_types = hyp_type_scores.max(dim=1)
            hypothesis_scores[:num_hyp] = hypothesis_scores[:num_hyp] + hyp_type_scores

            for i in range(num_hyp):
                base_id = base_index[i]
                new_stacked_types[i] = stacked_types[base_id]
                #new_stacked_types[i, t] = hyp_types[i]
                new_stacked_types[i, head-1] = hyp_types[i]

            stacked_heads = [[new_stacked_heads[i][j] for j in range(len(new_stacked_heads[i]))] for i in range(num_hyp)]
            arcs = [set(new_arcs[i]) for i in range(num_hyp)]
            positions = [new_positions[i] for i in range(num_hyp)]
            if self.grandPar:
                grand_parents = [[new_grand_parents[i][j] for j in range(len(new_grand_parents[i]))] for i in range(num_hyp)]
            if self.sibling:
                siblings = [[new_siblings[i][j] for j in range(len(new_siblings[i]))] for i in range(num_hyp)]
            if self.skipConnect:
                skip_connects = [[new_skip_connects[i][j] for j in range(len(new_skip_connects[i]))] for i in range(num_hyp)]
            #constraints = new_constraints

            #child_orders = new_child_orders
            children.copy_(new_children)
            stacked_types.copy_(new_stacked_types)
            # hx [decoder_layers, num_hyp, hidden_size]
            # hack to handle LSTM
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, base_index, :]
                cx = cx[:, base_index, :]
                hx = (hx, cx)
            else:
                hx = hx[:, base_index, :]

        children = children.cpu().numpy()[0]
        stacked_types = stacked_types.cpu().numpy()[0]
        heads = np.zeros(length, dtype=np.int32)
        types = np.zeros(length, dtype=np.int32)
        for i in range(num_step):
            #child = stack[-1]
            head = children[i]
            type = stacked_types[i]

            heads[i+1] = head
            types[i+1] = type

        if debug: exit(0)
        return heads, types, length, children, stacked_types

    def decode(self, input_word, input_char, input_pos, input_xpos, mask=None, length=None, hx=None, beam=1, leading_symbolic=0, ordered=True):
        # reset noise for decoder
        self.decoder.reset_noise(0) 

        # output from encoder [batch, length_encoder, tag_space]
        # output_enc [batch, length, input_size]
        # arc_c [batch, length, arc_space]
        # type_c [batch, length, type_space]
        # hn [num_direction, batch, hidden_size]
        output_enc, hn, mask, length = self._get_encoder_output(input_word, input_char, input_pos, input_xpos, mask_e=mask, length_e=length, hx=hx)
        # output size [batch, length_encoder, arc_space]
        arc_c = F.elu(self.arc_c(output_enc))
        # output size [batch, length_encoder, type_space]
        type_c = F.elu(self.type_c(output_enc))
        # [decoder_layers, batch, hidden_size
        hn = self._transform_decoder_init_state(hn)
        batch, max_len_e, _ = output_enc.size()

        heads = np.zeros([batch, max_len_e], dtype=np.int32)
        types = np.zeros([batch, max_len_e], dtype=np.int32)

        #children = np.zeros([batch, 2 * max_len_e - 1], dtype=np.int32)
        #stack_types = np.zeros([batch, 2 * max_len_e - 1], dtype=np.int32)

        children = np.zeros([batch, max_len_e - 1], dtype=np.int32)
        stack_types = np.zeros([batch, max_len_e - 1], dtype=np.int32)

        for b in range(batch):
            sent_len = None if length is None else length[b]
            # hack to handle LSTM
            if isinstance(hn, tuple):
                hx, cx = hn
                hx = hx[:, b, :].contiguous()
                cx = cx[:, b, :].contiguous()
                hx = (hx, cx)
            else:
                hx = hn[:, b, :].contiguous()

            preds = self._decode_per_sentence(output_enc[b], arc_c[b], type_c[b], hx, sent_len, beam, ordered, leading_symbolic)
            if preds is None:
                preds = self._decode_per_sentence(output_enc[b], arc_c[b], type_c[b], hx, sent_len, beam, False, leading_symbolic)
            hids, tids, sent_len, chids, stids = preds
            heads[b, :sent_len] = hids
            types[b, :sent_len] = tids

            #children[b, :2 * sent_len - 1] = chids
            #stack_types[b, :2 * sent_len - 1] = stids

            children[b, : sent_len - 1] = chids
            stack_types[b, : sent_len - 1] = stids

        return heads, types, children, stack_types

