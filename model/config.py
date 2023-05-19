from collections import namedtuple
import torch

HParams = namedtuple('HParams',
    'vocab_size, pad_idx, bos_idx,'
    'emb_size, hidden_size, context_size, latent_size, factor_emb_size,'
    'n_class1, n_class2, key_len, sens_num, sen_len, po_len,'
    'batch_size, drop_ratio, weight_decay, clip_grad_norm,'

    'max_lr, min_lr, warmup_steps, ndis,'
    'min_tr, burn_down_tr, decay_tr,'
    'tau_annealing_steps, min_tau,'
    'rec_warm_steps,noise_decay_steps,'

    'log_steps, sample_num, max_epoches,'
    'save_epoches, validate_epoches,'
    'fbatch_size, fmax_epoches, fsave_epoches,'
    'vocab_path, ivocab_path, train_data, valid_data,'
    'model_dir, data_dir, train_log_path, valid_log_path,'
    'fig_log_path,'

    'corrupt_ratio, dae_epoches, dae_batch_size,'
    'dae_max_lr, dae_min_lr, dae_warmup_steps,'
    'dae_min_tr, dae_burn_down_tr, dae_decay_tr,'

    'dae_log_steps, dae_validate_epoches, dae_save_epoches,'
    'dae_train_log_path, dae_valid_log_path,'

    'cl_batch_size, cl_epoches,'
    'cl_max_lr, cl_min_lr, cl_warmup_steps,'
    'cl_log_steps, cl_validate_epoches,'
    'cl_save_epoches, cl_train_log_path,'
    'cl_valid_log_path'
)


hparams = HParams(
    # --------------------
    # general settings
    vocab_size=-1, pad_idx=-1, bos_idx=-1,  # to be replaced by true size after loading dictionary
    emb_size=512, hidden_size=512, context_size=512,
    latent_size=256, factor_emb_size=64,

    # 因素1, 电影内容
    #   0: 消极, 1: 积极, 2: 无
    # 因素2, 电影人物
    #   0: 消极, 1:积极, 2: 无
    n_class1=3, n_class2=3,

    # 每个关键词两个字组成
    # sens_num由输入决定，或者随机。
    key_len=4, sens_num=1,
    sen_len=9, po_len=30,

    drop_ratio=0.15, weight_decay=2.5e-4, clip_grad_norm=2.0,

    vocab_path="../data/vocab.pickle",
    ivocab_path="../data/ivocab.pickle",
    train_data="../data/train_short.pickle",
    valid_data="../data/test_short.pickle",
    model_dir="../checkpoint/",
    data_dir="../data/",

    #--------------------------
    # review的设置
    batch_size=2, ndis=3,
    max_lr=8e-4, min_lr=5e-8, warmup_steps=6000, # learning rate decay
    min_tr=0.85, burn_down_tr=3, decay_tr=6, # epoches for teach forcing ratio decay
    tau_annealing_steps=6000, min_tau=0.01,# Gumbel temperature, from 1 to min_tau
    rec_warm_steps=1500, noise_decay_steps=8500,

    log_steps=200, sample_num=1, max_epoches=12,
    save_epoches=3, validate_epoches=1,

    # 用以对标记数据进行微调
    fbatch_size=8, fmax_epoches=3, fsave_epoches=1,

    train_log_path="../log/mix_train_log.txt",
    valid_log_path="../log/mix_valid_log.txt",
    fig_log_path="../log/",

    #-------------------------------------------
    # 预训练, dae设置
    dae_batch_size=8,
    corrupt_ratio=0.1,
    dae_max_lr=8e-4, dae_min_lr=5e-8, dae_warmup_steps=4500,
    dae_min_tr=0.85, dae_burn_down_tr=2, dae_decay_tr=6,  # epoches for teach forcing ratio decay
    dae_epoches=10,

    dae_log_steps=300,
    dae_validate_epoches=1,
    dae_save_epoches=2,
    dae_train_log_path="../log/dae_train_log.txt",
    dae_valid_log_path="../log/dae_valid_log.txt",
    # ------------------------------------------
    # 预训练, 分类器设置
    cl_batch_size=64,
    cl_max_lr=8e-4, cl_min_lr=5e-8, cl_warmup_steps=800,
    cl_epoches=10,

    cl_log_steps=100,
    cl_validate_epoches=1,
    cl_save_epoches=2,
    cl_train_log_path="../log/cl_train_log.txt",
    cl_valid_log_path="../log/cl_valid_log.txt",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")