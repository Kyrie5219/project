from vaegan import VAEGAN
from tool import Tool
from config import device, hparams
import utils

from dae_trainer import DAETrainer
from cl_trainer import ClassifierTrainer
from vg_trainer import VGTrainer


def pretrain(review, tool, hps):
    dae_trainer = DAETrainer(hps)
    cl_trainer = ClassifierTrainer(hps)

    # --------------------------------------
    print("dae pretraining...")
    dae_trainer.train(review, tool)
    print("dae pretraining done!")

    # ---------------------------------------
    print("classifier1 pretraining...")
    cl_trainer.train(review, tool, factor_id=1)
    print("classifier1 pretraining done!")

    print("classifier2 pretraining...")
    cl_trainer.train(review, tool, factor_id=2)
    print("classifier2 pretraining done!")
    # --------------------------------------


def train(review, tool, hps):
    last_epoch = utils.restore_checkpoint(hps.model_dir, device, review)

    if last_epoch is not None:
        print("检查点存在!直接恢复!")
    else:
        print("检查点不存在!从零开始训练!")

    vg_trainer = VGTrainer(hps)
    vg_trainer.train(review, tool)


def main():
    hps = hparams
    #key_len=4, sens_num=1,sen_len=9, po_len=30,corrupt_ratio=0.1;
    tool = Tool(hps.sens_num, hps.key_len,
        hps.sen_len, hps.poem_len, hps.corrupt_ratio)
    tool.load_dic(hps.vocab_path, hps.ivocab_path)
    vocab_size = tool.get_vocab_size()
    PAD_ID = tool.get_PAD_ID()
    B_ID = tool.get_B_ID()
    assert vocab_size > 0 and PAD_ID >= 0 and B_ID >= 0
    hps = hps._replace(vocab_size=vocab_size, pad_idx=PAD_ID, bos_idx=B_ID)

    print("超参数:")
    print(hps)
    input("请检查超参数，然后按任意键继续 >")

    review = VAEGAN(hps)
    review = review.to(device)

    # pretrain(review, tool, hps)
    train(review, tool, hps)




if __name__ == "__main__":
    main()



