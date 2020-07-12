# !/bin/bash
python3 main.py --g_pre_epoch=1 --d_pre_epoch=1 --adver_epoch=2 --mode="train_generator"
python3 main.py --g_pre_epoch=1 --d_pre_epoch=1 --adver_epoch=2 --mode="train_discriminator"
python3 main.py --g_pre_epoch=1 --d_pre_epoch=1 --adver_epoch=2 --mode="adversarial_train"