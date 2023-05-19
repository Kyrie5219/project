import numpy as np
import random

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from modules import LossWrapper, ScheduledOptim, ExponentialDecay
from logger import DAELogger
from config import device
import utils

class DAETrainer(object):

    def __init__(self, hps):
        self.hps = hps

    def run_validation(self, epoch, review, tool, lr):
        logger = DAELogger('valid')
        logger.set_batch_num(tool.valid_batch_num)
        logger.set_log_path(self.hps.dae_valid_log_path)
        logger.set_rate('learning_rate', lr)
        logger.set_rate('teach_ratio', review.get_teach_ratio())

        for step in range(0, tool.valid_batch_num):

            batch = tool.valid_batches[step]

            batch_keys = batch[0].to(device)
            batch_pos = batch[1].to(device)
            batch_dec_inps = [dec_inp.to(device) for dec_inp in batch[2]]
            # batch_lengths = batch[3].to(device)

            gen_loss, _ = self.run_step(review, None,
                batch_keys, batch_pos, batch_dec_inps, True)
            logger.add_losses(gen_loss)

        logger.print_log(epoch)


    def run_step(self, review, optimizer, keys, moviereview, dec_inps, valid=False):
        if not valid:
            optimizer.zero_grad()
        all_outs = \
            review.dae_graph(keys, moviereview, dec_inps)

        gen_loss = self.losswrapper.cross_entropy_loss(all_outs, dec_inps)

        if not valid:
            gen_loss.backward()
            clip_grad_norm_(review.dae_parameters(), self.hps.clip_grad_norm)
            optimizer.step()

        return gen_loss.item(), all_outs


    def run_train(self, review, tool, optimizer, logger):
        logger.set_start_time()

        for step in range(0, tool.train_batch_num):

            batch = tool.train_batches[step]
            batch_keys = batch[0].to(device)
            batch_reviews = batch[1].to(device)
            batch_dec_inps = [dec_inp.to(device) for dec_inp in batch[2]]
            # batch_lengths = batch[3].to(device)
            # print('batch_dec_inps', batch_dec_inps)
            gen_loss, outs = \
                self.run_step(review, optimizer,
                    batch_keys, batch_reviews, batch_dec_inps)

            logger.add_losses(gen_loss)
            logger.set_rate("学习率", optimizer.rate())
            if step % self.hps.dae_log_steps == 0:
                logger.set_end_time()

                utils.sample_dae(batch_keys, batch_reviews, batch_dec_inps,
                    outs, self.hps.sample_num, tool)
                logger.print_log()
                logger.set_start_time()


    def train(self, review, tool):
        utils.print_parameter_list(review, review.dae_parameter_names())
        #input("please check the parameters, and then press any key to continue >")

        # 加载预训练数据
        print("为dae构建数据...")
        tool.build_data(self.hps.train_data, self.hps.valid_data,
            self.hps.dae_batch_size, mode='dae')

        print("train batch num: %d" %(tool.train_batch_num))
        print("valid batch num: %d" %(tool.valid_batch_num))


        # 训练logger
        logger = DAELogger('train')
        logger.set_batch_num(tool.train_batch_num)
        logger.set_log_steps(self.hps.dae_log_steps)
        logger.set_log_path(self.hps.dae_train_log_path)
        logger.set_rate('learning_rate', 0.0)
        logger.set_rate('teach_ratio', 1.0)


        # 构建优化器optimizer
        opt = torch.optim.AdamW(review.dae_parameters(),
            lr=1e-3, betas=(0.9, 0.99), weight_decay=self.hps.weight_decay)
        optimizer = ScheduledOptim(optimizer=opt, warmup_steps=self.hps.dae_warmup_steps,
            max_lr=self.hps.dae_max_lr, min_lr=self.hps.dae_min_lr)

        review.train()

        self.losswrapper = LossWrapper(pad_idx=tool.get_PAD_ID(), sens_num=self.hps.sens_num,
            sen_len=self.hps.sen_len)

        # tech forcing ratio decay衰变
        tr_decay_tool = ExponentialDecay(self.hps.dae_burn_down_tr, self.hps.dae_decay_tr,
            self.hps.dae_min_tr)

        # 训练
        for epoch in range(1, self.hps.dae_epoches+1):

            self.run_train(review, tool, optimizer, logger)

            if epoch % self.hps.dae_validate_epoches == 0:
                print("run validation...")
                review.eval()
                print("in training mode: %d" % (review.training))
                self.run_validation(epoch, review, tool, optimizer.rate())
                review.train()
                print("validation Done: %d" % (review.training))


            if (self.hps.dae_save_epoches >= 1) and \
                (epoch % self.hps.dae_save_epoches) == 0:
                # save checkpoint
                print("保存模型...")
                utils.save_checkpoint(self.hps.model_dir, epoch, review, prefix="dae")


            logger.add_epoch()

            print("teach forcing ratio decay...")
            review.set_teach_ratio(tr_decay_tool.do_step())
            logger.set_rate('teach_ratio', tr_decay_tool.get_rate())

            print("shuffle data...")
            tool.shuffle_train_data()
