import numpy as np
import random

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from modules import LossWrapper, ScheduledOptim, ExponentialDecay, LinearDecay
from config import device
from logger import VGLogger
import utils


class VGTrainer(object):

    def __init__(self, hps):
        self.hps = hps


    def run_validation(self, epoch, review, tool, lr):
        logger = VGLogger('valid')
        logger.set_batch_num(tool.valid_batch_num)
        logger.set_log_path(self.hps.valid_log_path, "")
        logger.set_rate('learning_rate', lr)
        logger.set_rate('teach_ratio', review.get_teach_ratio())
        logger.set_rate('noise_weight', self.noise_decay_tool.get_rate())

        for step in range(0, tool.valid_batch_num):

            batch = tool.valid_batches[step]

            batch_keys = batch[0].to(device)
            batch_pos = batch[1].to(device)
            batch_dec_inps = [dec_inp.to(device) for dec_inp in batch[2]]
            batch_labels = batch[3].to(device)
            batch_label_mask = batch[4].to(device)
            # batch_lengths = batch[5].to(device)

            rec_loss, cl_loss_w, cl_loss_xw, entro_loss, _, _, _ = \
                self.run_rec_step(review, None, batch_keys, batch_pos,
                    batch_dec_inps, batch_labels, batch_label_mask, True)

            logger.add_rec_losses(rec_loss, cl_loss_w, cl_loss_xw, entro_loss)

            dis_loss = self.run_dis_step(review, None, batch_keys,
                batch_pos, batch_labels, batch_label_mask, True)

            logger.add_dis_loss(dis_loss)

            adv_loss = self.run_adv_step(review, None, batch_keys,
                batch_pos, batch_labels, batch_label_mask, True)

            logger.add_adv_loss(adv_loss)


        logger.print_log(epoch)


    def run_adv_step(self, review, optimizer, keys, moviereview,
        labels, label_mask, valid=False):
        if not valid:
            optimizer.zero_grad()


        z_prior, z_post, cb_label1, cb_label2 \
                = review.get_prior_and_posterior(keys, moviereview, labels, label_mask)

        noise_weight = self.noise_decay_tool.get_rate()

        eps1 = torch.randn_like(z_post) * noise_weight
        eps2 = torch.randn_like(z_prior) * noise_weight

        logits_real = review.layers['discriminator'](z_post+eps1, cb_label1, cb_label2)
        logits_fake = review.layers['discriminator'](z_prior+eps2, cb_label1, cb_label2)

        # RSGAN
        loss = F.logsigmoid(logits_fake-logits_real)
        loss = -loss.mean()

        if not valid:
            loss.backward()
            clip_grad_norm_(review.gen_parameters(), self.hps.clip_grad_norm)
            optimizer.step()


        return loss.item()


    def run_dis_step(self, review, optimizer, keys, moviereview,
        labels, label_mask, valid=False):
        if not valid:
            optimizer.zero_grad()

        with torch.no_grad():
            z_prior, z_post, cb_label1, cb_label2 \
                = review.get_prior_and_posterior(keys, moviereview, labels, label_mask)


        noise_weight = self.noise_decay_tool.get_rate()

        eps1 = torch.randn_like(z_post) * noise_weight
        eps2 = torch.randn_like(z_prior) * noise_weight

        logits_real = review.layers['discriminator'](z_post+eps1, cb_label1, cb_label2)
        logits_fake = review.layers['discriminator'](z_prior+eps2, cb_label1, cb_label2)

        # RSGAN
        loss = F.logsigmoid(logits_real-logits_fake)
        loss = -loss.mean()


        if not valid:
            loss.backward()
            clip_grad_norm_(review.dis_parameters(), self.hps.clip_grad_norm)
            optimizer.step()

        return loss.item()


    def run_rec_step(self, review, optimizer, keys, moviereview, dec_inps,
        labels, label_mask, valid=False):

        if not valid:
            optimizer.zero_grad()

        all_outs, combined_label1, combined_label2, \
            logits_cl_xw1, logits_cl_xw2, logits_cl_w1, logits_cl_w2, = \
            review(keys, moviereview, dec_inps, labels, label_mask)

        rec_loss = self.losswrapper.cross_entropy_loss(all_outs, dec_inps)

        cl_loss_w1, cl_loss_xw1, entro_loss_xw1 = self.losswrapper.cl_loss(logits_cl_w1,
            logits_cl_xw1, combined_label1, label_mask[:, 0])
        cl_loss_w2, cl_loss_xw2, entro_loss_xw2 = self.losswrapper.cl_loss(logits_cl_w2,
            logits_cl_xw2, combined_label2, label_mask[:, 1])

        cl_loss_w = cl_loss_w1 + cl_loss_w2
        cl_loss_xw = cl_loss_xw1 + cl_loss_xw2
        entro_loss_xw = entro_loss_xw1 + entro_loss_xw2


        loss = rec_loss + cl_loss_w + cl_loss_xw + entro_loss_xw


        if not valid:
            loss.backward()
            clip_grad_norm_(review.rec_parameters(), self.hps.clip_grad_norm)
            optimizer.step()

        return rec_loss.item(), cl_loss_w.item(), cl_loss_xw.item(),\
            entro_loss_xw.item(), all_outs, combined_label1, combined_label2

    # ----------------------------------------------------
    def gen_from_prior(self, review, keys, moviereview, dec_inps,
            labels, label_mask):

        with torch.no_grad():

            all_outs, _, _, _, _, _, _, = \
                review(keys, moviereview, dec_inps, labels, label_mask, True, 0.0)

        return all_outs

    # -------------------------------------------------------------------------
    def run_train(self, review, tool, optimizerRec, optimizerDis,
                optimizerGen, logger, epoch):

        logger.set_start_time()

        for step in range(0, tool.train_batch_num):

            batch = tool.train_batches[step]
            batch_keys = batch[0].to(device)
            batch_reviews = batch[1].to(device)
            batch_dec_inps = [dec_inp.to(device) for dec_inp in batch[2]]
            batch_labels = batch[3].to(device)
            batch_label_mask = batch[4].to(device)
            # batch_lengths = batch[5].to(device)

            # train the classifier, recognition network and decoder
            rec_loss, cl_loss_w, cl_loss_xw, entro_loss, outs_post, clabels1, clabels2 = \
                self.run_rec_step(review, optimizerRec,
                    batch_keys, batch_reviews, batch_dec_inps,
                    batch_labels, batch_label_mask)

            logger.add_rec_losses(rec_loss, cl_loss_w, cl_loss_xw, entro_loss)
            logger.set_rate("learning_rate", optimizerRec.rate())


            # train discriminator
            if logger.total_steps > self.hps.rec_warm_steps:
                dis_loss = 0
                for i in range(0, self.hps.ndis):
                    step_dis_loss = self.run_dis_step(review, optimizerDis,
                        batch_keys, batch_reviews, batch_labels, batch_label_mask)
                    dis_loss += step_dis_loss
                dis_loss /= self.hps.ndis
                logger.add_dis_loss(dis_loss)
                logger.set_rate('noise_weight', self.noise_decay_tool.do_step())


            if logger.total_steps > self.hps.rec_warm_steps:
                # train prior and posterior generators
                adv_loss = self.run_adv_step(review, optimizerGen,
                    batch_keys, batch_reviews, batch_labels, batch_label_mask)

                logger.add_adv_loss(adv_loss)


            # temperature annealing
            review.set_tau(self.tau_decay_tool.do_step())
            logger.set_rate('temperature', self.tau_decay_tool.get_rate())

            if (step % 40 == 0) and (logger.total_steps > self.hps.rec_warm_steps):
                dist = utils.cal_distance(review, batch_keys, batch_reviews,
                    batch_labels, batch_label_mask)
                if not np.isnan(dist):
                    logger.add_distance(dist)
                #------------
                fadist = utils.factor_distance(review, batch_keys,
                    self.hps.n_class1, self.hps.n_class2, device)
                if not np.isnan(fadist):
                    logger.add_factor_distance(fadist)


            if step % self.hps.log_steps == 0:
                logger.set_end_time()
                outs_prior = self.gen_from_prior(review,
                    batch_keys, batch_reviews, batch_dec_inps,
                    batch_labels, batch_label_mask)

                utils.sample_mix(batch_keys, batch_dec_inps, batch_labels,
                    clabels1, clabels2, outs_post, outs_prior, self.hps.sample_num, tool)
                logger.print_log()
                logger.draw_curves()
                logger.set_start_time()


    def train(self, review, tool):

        print("building data for vaegan...")
        tool.build_data(self.hps.train_data, self.hps.valid_data,
            self.hps.batch_size, mode='review_pre')

        print("train batch num: %d" % (tool.train_batch_num))
        print("valid batch num: %d" % (tool.valid_batch_num))


        # training logger
        logger = VGLogger('train')
        logger.set_batch_num(tool.train_batch_num)
        logger.set_log_steps(self.hps.log_steps)
        logger.set_log_path(self.hps.train_log_path, self.hps.fig_log_path)
        logger.set_rate('learning_rate', 0.0)
        logger.set_rate('teach_ratio', 1.0)
        logger.set_rate('temperature', 1.0)


        # build optimizer
        optRec = torch.optim.AdamW(review.rec_parameters(),
            lr=1e-3, betas=(0.9, 0.99), weight_decay=self.hps.weight_decay)
        optimizerRec = ScheduledOptim(optimizer=optRec, warmup_steps=self.hps.warmup_steps,
            max_lr=self.hps.max_lr, min_lr=self.hps.min_lr)

        optDis = torch.optim.AdamW(review.dis_parameters(),
            lr=1e-3, betas=(0.5, 0.99), weight_decay=self.hps.weight_decay)
        optimizerDis = ScheduledOptim(optimizer=optDis, warmup_steps=self.hps.warmup_steps,
            max_lr=self.hps.max_lr, min_lr=self.hps.min_lr, beta=0.5)

        optGen = torch.optim.AdamW(review.gen_parameters(),
            lr=1e-3, betas=(0.9, 0.99), weight_decay=self.hps.weight_decay)
        optimizerGen = ScheduledOptim(optimizer=optGen, warmup_steps=self.hps.warmup_steps,
            max_lr=self.hps.max_lr, min_lr=self.hps.min_lr, beta=0.55)

        # set loggers
        review.train()

        self.losswrapper = LossWrapper(pad_idx=tool.get_PAD_ID(),
            sens_num=self.hps.sens_num, sen_len=self.hps.sen_len)
        # change each epoch
        tr_decay_tool = ExponentialDecay(self.hps.burn_down_tr, self.hps.decay_tr, self.hps.min_tr)
        # change each iteration
        self.tau_decay_tool = ExponentialDecay(0, self.hps.tau_annealing_steps, self.hps.min_tau)
        self.noise_decay_tool = LinearDecay(0, self.hps.noise_decay_steps, max_v=0.3, min_v=0.0)


        # -----------------------------------------------------------
        # train with all data
        for epoch in range(1, self.hps.max_epoches+1):

            self.run_train(review, tool, optimizerRec, optimizerDis,
                optimizerGen, logger, epoch)

            if epoch % self.hps.validate_epoches == 0:
                print("run validation...")
                review.eval()
                print ("in training mode: %d" % (review.training))
                self.run_validation(epoch, review, tool, optimizerRec.rate())
                review.train()
                print ("validation Done: %d" % (review.training))


            if (self.hps.save_epoches >= 0) and \
                (epoch % self.hps.save_epoches) == 0:
                # save checkpoint
                print("saving model...")
                utils.save_checkpoint(self.hps.model_dir, epoch, review, "mixpre",
                    optimizerRec, optimizerDis, optimizerGen)

            logger.add_epoch()

            print("teach forcing ratio decay...")
            review.set_teach_ratio(tr_decay_tool.do_step())
            logger.set_rate('teach_ratio', tr_decay_tool.get_rate())

            print("shuffle data...")
            tool.shuffle_train_data()

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        # fine-tune with only labelled data
        print("building data for fine-tuning ...")
        tool.build_data(self.hps.train_data, self.hps.valid_data,
            self.hps.fbatch_size, mode='review_tune')

        print("train batch num: %d" % (tool.train_batch_num))
        print("valid batch num: %d" % (tool.valid_batch_num))


        # training logger
        logger.set_batch_num(tool.train_batch_num)


        for epoch in range(1, self.hps.fmax_epoches+1):

            self.run_train(review, tool, optimizerRec, optimizerDis,
                optimizerGen, logger, epoch)

            if epoch % self.hps.validate_epoches == 0:
                print("run validation...")
                review.eval()
                print("in training mode: %d" % (review.training))
                self.run_validation(epoch, review, tool, optimizerRec.rate())
                review.train()
                print("validation Done: %d" % (review.training))


            if (self.hps.fsave_epoches >= 0) and \
                (epoch % self.hps.fsave_epoches) == 0:
                # save checkpoint
                print("saving model...")
                utils.save_checkpoint(self.hps.model_dir, epoch, review, "mixfine",
                    optimizerRec, optimizerDis, optimizerGen)

            logger.add_epoch()

            print("teach forcing ratio decay...")
            review.set_teach_ratio(tr_decay_tool.do_step())
            logger.set_rate('teach_ratio', tr_decay_tool.get_rate())

            print("shuffle data...")
            tool.shuffle_train_data()