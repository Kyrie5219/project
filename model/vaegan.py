import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from itertools import chain
import numpy as np
from math import ceil, log2
from modules import MLP, PriorGenerator, PosterioriGenerator, get_non_pad_mask, get_seq_length, GumbelSampler, ContextLayer
from spectral_normalization import SpectralNorm as SN
from transformers import GPT2Model
from trm_modules import Trm_encoder, PositionalEncoding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#VAE-GAN结构
#encoder
class VGencoder(nn.Module):
    def __init__(self, input_size, hidden_size, cell='GRU', n_layers=1, drop_ratio=0.1):
        super(VGencoder, self).__init__()
        self.cell_type = cell
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        if cell == 'GRU':
            self.deepnet = nn.GRU(input_size, hidden_size, n_layers, bidirectional=True, batch_first=True)
        elif cell == 'Elman':
            self.deepnet = nn.RNN(input_size, hidden_size, n_layers, bidirectional=True, batch_first=True)
        elif cell == 'Trm':
            self.deepnet = Trm_encoder(position_encoder=PositionalEncoding, input_size=input_size)

        self.dropout_layer = nn.Dropout(drop_ratio)

    def forward(self, embed_seq, input_lens=None):
        # embed_seq: (emb_dim)
        embed_inps = self.dropout_layer(embed_seq)

        if input_lens == None:
            outputs, state = self.deepnet(embed_inps, None)
        else:
            total_len = embed_inps.size(1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(embed_inps,
                                                             input_lens, batch_first=True, enforce_sorted=False)
            outputs, state = self.deepnet(packed, None)
            # outputs: (B, L, num_directions*H)
            # state: (num_layers*num_directions, B, H)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                                                                batch_first=True, total_length=total_len)
        return outputs, state
    # bi-direction
    def init_state(self, batch_size):
        init_h = torch.zeros( (self.n_layers*2, batch_size,
            self.hidden_size), requires_grad=False, device=device)

        if self.cell_type == 'LSTM':
            init_c = torch.zeros((self.n_layers*2, batch_size,
                self.hidden_size), requires_grad=False, device=device)
            return (init_h, init_c)
        else:
            return init_h

#decoder
class VGdecoder(nn.Module):
    def __init__(self, input_size, hidden_size, cell='Trm', n_layers=1, drop_ratio=0.1):
        super(VGdecoder, self).__init__()
        self.dropout_layer = nn.Dropout(drop_ratio)
        self.cell = cell
        if cell == 'GRU':
            self.deepnet = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
        elif cell == 'Trm':
            self.deepnet = GPT2Model.from_pretrained("../data/Trm_path")

    def forward(self, embed_seq, last_state, past_key_values=None):
        embed_inps = self.dropout_layer(embed_seq)
        if self.cell != 'Trm':
            output, state = self.deepnet(embed_inps, last_state.unsqueeze(0))
            output = output.squeeze(1)
            return output, state.squeeze(0)
        else:
            # print('embed_inps', embed_inps.size())
            output = self.deepnet(inputs_embeds=embed_inps, past_key_values=past_key_values)
            # print('output', output.shape())
            return output

#鉴别器
class Discriminator(nn.Module):
    def __init__(self, n_class1, n_class2, factor_emb_size,
        latent_size, drop_ratio):
        super(Discriminator, self).__init__()
        # (B, L, H)
        self.inp2feature = nn.Sequential(
            SN(nn.Linear(latent_size, latent_size)),
            nn.LeakyReLU(0.2),
            SN(nn.Linear(latent_size, factor_emb_size*2)),
            nn.LeakyReLU(0.2),
            SN(nn.Linear(factor_emb_size*2, factor_emb_size*2)),
            nn.LeakyReLU(0.2))
        self.feature2logits = SN(nn.Linear(factor_emb_size*2, 1))

        # 鉴别器factor embedding
        self.dis_fembed1 = SN(nn.Embedding(n_class1, factor_emb_size))
        self.dis_fembed2 = SN(nn.Embedding(n_class2, factor_emb_size))

    def forward(self, x, labels1, labels2):
        # x, latent variable: (B, latent_size)

        femb1 = self.dis_fembed1(labels1)
        femb2 = self.dis_fembed2(labels2)
        factor_emb = torch.cat([femb1, femb2], dim=1) # (B, factor_emb_size*2)

        feature = self.inp2feature(x)  # (B, factor_emb_size*2)
        logits0 = self.feature2logits(feature).squeeze(1)  # (B)

        logits = logits0 + torch.sum(feature*factor_emb, dim=1)

        return logits


class VAEGAN(nn.Module):
    def __init__(self, hps):
        super(VAEGAN, self).__init__()
        self.hps = hps

        self.vocab_size = hps.vocab_size
        self.n_class1 = hps.n_class1
        self.n_class2 = hps.n_class2
        self.emb_size = hps.emb_size
        self.hidden_size = hps.hidden_size
        self.factor_emb_size = hps.factor_emb_size
        self.latent_size = hps.latent_size
        self.context_size = hps.context_size
        self.po_len = hps.po_len
        self.sens_num = hps.sens_num
        self.sen_len = hps.sen_len

        self.pad_idx = hps.pad_idx
        self.bos_idx = hps.bos_idx

        self.bos_tensor = torch.tensor(hps.bos_idx, dtype=torch.long, device=device).view(1, 1)

        self.gumbel_tool = GumbelSampler()

        # 位置补偿
        self.pos_inps = F.one_hot(torch.arange(0, self.sens_num), self.sens_num)
        self.pos_inps = self.pos_inps.type(torch.FloatTensor).to(device)

        self.layers = nn.ModuleDict()
        self.layers['embed'] = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=self.pad_idx)
        self.layers['encoder'] = VGencoder(self.emb_size, self.hidden_size, drop_ratio=hps.drop_ratio)

        self.layers['decoder'] = VGdecoder(self.hidden_size, self.hidden_size, drop_ratio=hps.drop_ratio)

        self.layers['word_encoder'] = VGencoder(self.emb_size, self.emb_size, cell='Elman',
            drop_ratio=hps.drop_ratio)

        # p(y_1|x,w), p(y_2|x,w)
        self.layers['cl_xw1'] = MLP(self.hidden_size*2+self.emb_size*2,
            layer_sizes=[self.hidden_size, 128, self.n_class1], activs=['relu', 'relu', None],
            drop_ratio=hps.drop_ratio)
        self.layers['cl_xw2'] = MLP(self.hidden_size*2+self.emb_size*2,
            layer_sizes=[self.hidden_size, 128, self.n_class2], activs=['relu', 'relu', None],
            drop_ratio=hps.drop_ratio)

        # p(y_1|w), p(y_2|w)
        self.layers['cl_w1'] = MLP(self.emb_size*2,
            layer_sizes=[self.emb_size, 64, self.n_class1], activs=['relu', 'relu', None],
            drop_ratio=hps.drop_ratio)
        self.layers['cl_w2'] = MLP(self.emb_size*2,
            layer_sizes=[self.emb_size, 64, self.n_class2], activs=['relu', 'relu', None],
            drop_ratio=hps.drop_ratio)

        # factor embedding
        self.layers['factor_embed1'] = nn.Embedding(self.n_class1, self.factor_emb_size)
        self.layers['factor_embed2'] = nn.Embedding(self.n_class2, self.factor_emb_size)

        # 后验与先验
        self.layers['prior'] = PriorGenerator(
            self.emb_size*2+int(self.latent_size//2),
            self.latent_size, self.n_class1, self.n_class2, self.factor_emb_size)

        self.layers['posteriori'] = PosterioriGenerator(
            self.hidden_size*2+self.emb_size*2, self.latent_size,
            self.n_class1, self.n_class2, self.factor_emb_size)

        # 对抗训练鉴别器
        self.layers['discriminator'] = Discriminator(self.n_class1, self.n_class2,
            self.factor_emb_size, self.latent_size, drop_ratio=hps.drop_ratio)

        #解码器输出--词汇表输出
        # self.layers['out_proj'] = nn.Linear(hps.hidden_size, hps.vocab_size)
        self.layers['out_proj'] = nn.Linear(768, hps.vocab_size)
        #用于计算解码器初始状态的MLP。
        self.layers['dec_init'] = MLP(self.latent_size+self.emb_size*2+self.factor_emb_size*2,
            layer_sizes=[self.hidden_size],
            activs=['tanh'], drop_ratio=hps.drop_ratio)

        # self.layers['map_x'] = MLP(self.context_size+self.emb_size,
        self.layers['map_x'] = MLP(self.emb_size,
            # layer_sizes=[self.hidden_size],
            layer_sizes=[768],
            activs=['tanh'], drop_ratio=hps.drop_ratio)

        self.layers['context'] = ContextLayer(self.hidden_size, self.context_size)

        # two annealing parameters
        self.__tau = 1.0
        self.__teach_ratio = 1.0

        #预训练
        self.layers['dec_init_pre'] = MLP(self.hidden_size*2+self.emb_size*2,
            layer_sizes=[self.hidden_size],
            activs=['tanh'], drop_ratio=hps.drop_ratio)

        self.cell_type = 'Trm'

    def set_tau(self, tau):
        self.gumbel_tool.set_tau(tau)


    def get_tau(self):
        self.gumbel_tool.get_tau()


    def set_teach_ratio(self, teach_ratio):
        if 0.0 < teach_ratio <= 1.0:
            self.__teach_ratio = teach_ratio


    def get_teach_ratio(self):
        return self.__teach_ratio


    # def dec_step(self, inp, state, context):
    def dec_step(self, inp, state, past_key_values=None):
        emb_inp = self.layers['embed'](inp)

        # x = self.layers['map_x'](torch.cat([emb_inp, context.unsqueeze(1)], dim=2))
        x = self.layers['map_x'](emb_inp)
        if self.cell_type == 'Trm':
            cell_out = self.layers['decoder'](x, state, past_key_values)
            cell_out, past_key_values = cell_out[0], cell_out[1]
            out = self.layers['out_proj'](cell_out).squeeze(1)
            return out, past_key_values
        else:
            cell_out, new_state = self.layers['decoder'](x, state)
            out = self.layers['out_proj'](cell_out)
            return out, new_state


    def generator(self, dec_init_state, dec_inps, specified_teach=None):
        # the decoder p(x|z, w, y)
        # 初始化context向量
        batch_size = dec_init_state.size(0)
        # context = torch.zeros((batch_size, self.context_size), dtype=torch.float, device=device)  # (B, context_size)

        all_outs = []
        if specified_teach is None:
            teach_ratio = self.__teach_ratio
        else:
            teach_ratio = specified_teach

        for step in range(0, self.sens_num):

            state = dec_init_state
            max_dec_len = dec_inps[step].size(1)

            outs = torch.zeros(batch_size, max_dec_len, self.vocab_size, device=device)
            dec_states = []

            inp = self.bos_tensor.expand(batch_size, 1)
            past_key_values = None
            for t in range(0, max_dec_len):
                if self.cell_type != 'Trm':
                    out, state = self.dec_step(inp, state)
                    outs[:, t, :] = out
                else:
                    out, past_key_values = self.dec_step(inp, state, past_key_values)
                    outs[:, t, :] = out
                # teach force with a probability
                is_teach = random.random() < teach_ratio
                if is_teach or (not self.training):
                    inp = dec_inps[step][:, t].unsqueeze(1)
                else:
                    normed_out = F.softmax(out, dim=-1)
                    top1 = normed_out.data.max(1)[1]
                    inp = top1.unsqueeze(1)

                dec_states.append(state.unsqueeze(2))  # (B, H, 1)

            all_outs.append(outs)


        return all_outs


    def computer_enc(self, inps, encoder):
        lengths = get_seq_length(inps, self.pad_idx)

        emb_inps = self.layers['embed'](inps)  # (batch_size, length, emb_size)

        enc_outs, enc_state = encoder(emb_inps, lengths)
        return enc_outs, enc_state

    def get_prior_and_posterior(self, key_inps, vae_inps, factor_labels,
        factor_mask, ret_others=False):
        # get the representation of a whole review
        _, vae_state = self.computer_enc(vae_inps, self.layers['encoder'])  # (2, B, H)
        sen_state = torch.cat([vae_state[0, :, :], vae_state[1, :, :]], dim=-1)  # [B, 2*H]

        # get the representation of the keyword
        # TODO: incorporate multiple keywords
        _, key_state0 = self.computer_enc(key_inps, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]

        condition = torch.cat([sen_state, key_state], dim=1)
        # get embedding of either provided or sampled labels
        factor_emb1, logits_cl_xw1, combined_label1 = self.get_factor_emb(condition, 1,
            factor_labels[:, 0], factor_mask[:, 0])
        factor_emb2, logits_cl_xw2, combined_label2 = self.get_factor_emb(condition, 2,
            factor_labels[:, 1], factor_mask[:, 1])

        factors = torch.cat([factor_emb1, factor_emb2], dim=-1)


        # get posteriori p(z|x,w,y)
        batch_size = key_state.size(0)
        eps = torch.randn((batch_size, self.latent_size), dtype=torch.float, device=device)
        z_post = self.layers['posteriori'](sen_state, key_state, combined_label1, combined_label2)
        z_prior = self.layers['prior'](key_state, combined_label1, combined_label2)

        if ret_others:

            return z_prior, z_post, key_state, factors,\
                logits_cl_xw1, logits_cl_xw2, combined_label1, combined_label2
        else:
            return z_prior, z_post, combined_label1, combined_label2


    def forward(self, key_inps, vae_inps, dec_inps, factor_labels, factor_mask,
        use_prior=False, specified_teach=None):

        z_prior, z_post, key_state, factors, logits_cl_xw1, logits_cl_xw2, cb_label1, cb_label2\
            = self.get_prior_and_posterior(key_inps, vae_inps, factor_labels, factor_mask, True)

        if use_prior:
            z = z_prior
        else:
            z = z_post

        # generate lines
        dec_init_state = self.layers['dec_init'](torch.cat([z, key_state, factors], dim=-1))  # (B, H-2)
        all_gen_outs = self.generator(dec_init_state, dec_inps, specified_teach)

        # 分类损失
        logits_cl_w1 = self.layers['cl_w1'](key_state)
        logits_cl_w2 = self.layers['cl_w2'](key_state)

        return all_gen_outs, cb_label1, cb_label2, \
            logits_cl_xw1, logits_cl_xw2, logits_cl_w1, logits_cl_w2


    def dae_graph(self, keys, moviereview, dec_inps):
        # 预训练编码器和解码器作为去噪自动编码器
        # if self.cell_type == 'Trm':
        #    movie_state = self.computer_enc(moviereview, self.layers['encoder'])
        # else:
        _, movie_state0 = self.computer_enc(moviereview, self.layers['encoder'])
        movie_state = torch.cat([movie_state0[0, :, :], movie_state0[1, :, :]], dim=-1)  # [B, 2*H]

        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1)  # [B, 2*H]
        dec_init_state = self.layers['dec_init_pre'](torch.cat([movie_state, key_state], dim=-1))

        all_gen_outs = self.generator(dec_init_state, dec_inps)

        return all_gen_outs


    def classifier_graph(self, keys, reviews, factor_id):
        _, review_state0 = self.computer_enc(reviews, self.layers['encoder'])
        review_state = torch.cat([review_state0[0, :, :], review_state0[1, :, :]], dim=-1) # [B, 2*H]

        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1) # [B, 2*H]

        condition = torch.cat([review_state, key_state], dim=-1)

        logits_w = self.layers['cl_w'+str(factor_id)](key_state)
        logits_xw = self.layers['cl_xw'+str(factor_id)](condition)

        probs_w = F.softmax(logits_w, dim=-1)
        probs_xw = F.softmax(logits_xw, dim=-1)


        return logits_xw, logits_w, probs_xw, probs_w


    def dae_parameter_names(self):
        required_names = ['embed', 'encoder', 'word_encoder',
                          'dec_init_pre', 'decoder', 'out_proj', 'context', 'map_x']
        return required_names

    def dae_parameters(self):
        names = self.dae_parameter_names()

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)

    # -------------------------------------
    def classifier_parameter_names(self, factor_id):
        assert factor_id == 1 or factor_id == 2
        if factor_id == 1:
            required_names = ['embed', 'encoder', 'word_encoder',
                              'cl_w1', 'cl_xw1']
        else:
            required_names = ['embed', 'encoder', 'word_encoder',
                              'cl_w2', 'cl_xw2']
        return required_names

    def cl_parameters(self, factor_id):
        names = self.classifier_parameter_names(factor_id)

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)

    # ---------------------------------------------
    # for adversarial training
    def rec_parameters(self):
        # parameters of the classifiers, recognition network and decoder
        names = ['embed', 'encoder', 'decoder', 'word_encoder',
                 'cl_xw1', 'cl_xw2', 'cl_w1', 'cl_w2',
                 'factor_embed1', 'factor_embed2', 'posteriori',
                 'out_proj', 'dec_init', 'context', 'map_x']

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)

    def dis_parameters(self):
        # parameters of the discriminator
        return self.layers['discriminator'].parameters()

    def gen_parameters(self):
        # parameters of the recognition network and prior network
        names = ['prior', 'posteriori', 'encoder', 'word_encoder',
                 'embed']

        required_params = [self.layers[name].parameters() for name in names]

        return chain.from_iterable(required_params)

    # ------------------------------------------------
    # functions for generating
    def compute_key_state(self, keys):
        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1)  # [B, 2*H]
        return key_state

    def compute_inferred_label(self, key_state, factor_id):
        logits = self.layers['cl_w' + str(factor_id)](key_state)
        probs = F.softmax(logits, dim=-1)
        pred = probs.max(dim=-1)[1]  # (B)
        return pred

    def compute_dec_init_state(self, key_state, labels1, labels2):
        z_prior = self.layers['prior'](key_state, labels1, labels2)

        factor_emb1 = self.layers['factor_embed1'](labels1)
        factor_emb2 = self.layers['factor_embed2'](labels2)

        dec_init_state = self.layers['dec_init'](
            torch.cat([z_prior, key_state, factor_emb1, factor_emb2], dim=-1))  # (B, H-2)

        return dec_init_state

    def compute_prior(self, keys, labels1, labels2):
        _, key_state0 = self.computer_enc(keys, self.layers['word_encoder'])
        key_state = torch.cat([key_state0[0, :, :], key_state0[1, :, :]], dim=-1)  # [B, 2*H]

        z_prior = self.layers['prior'](key_state, labels1, labels2)

        return z_prior

    def get_factor_emb(self, condition, factor_id, label, mask):
        # -----------------------------------
        # sample labels for unlabelled poems from the classifier
        logits_cl = self.layers['cl_xw'+str(factor_id)](condition)
        sampled_label = self.gumbel_tool(logits_cl)

        fin_label = label.float() * mask + (1-mask) * sampled_label
        fin_label = fin_label.long()

        factor_emb = self.layers['factor_embed'+str(factor_id)](fin_label)

        return factor_emb, logits_cl, fin_label