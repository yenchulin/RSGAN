# !/bin/bash

# Please first mv this file outside DPGAN
# and copy DPGAN folder without any result naming DPGAN_copy


# epoch * 5/8
# both (No AE)
cd DPGAN_copy
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --auto_encoder=false --mode="train_generator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --auto_encoder=false --mode="train_discriminator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --auto_encoder=false --mode="adversarial_train"

cd ..
mkdir DPGAN_both_multi_doc_LDA_topic5_4
mv DPGAN_copy/myexperiment DPGAN_both_multi_doc_LDA_topic5_4
mv DPGAN_copy/test_max_generated DPGAN_both_multi_doc_LDA_topic5_4
mv DPGAN_copy/test_sample_generated DPGAN_both_multi_doc_LDA_topic5_4
mv DPGAN_copy/train_sample_generated DPGAN_both_multi_doc_LDA_topic5_4
mv DPGAN_copy/pretrain_test_sample_generated DPGAN_both_multi_doc_LDA_topic5_4
mv DPGAN_copy/MLE DPGAN_both_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_test DPGAN_both_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_train DPGAN_both_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_result DPGAN_both_multi_doc_LDA_topic5_4
rmdir -rf DPGAN_copy/__pycache__

# baseline (No AE)
cd DPGAN_copy
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --aspect_attn=false --auto_encoder=false --mode="train_generator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --aspect_attn=false --auto_encoder=false --mode="train_discriminator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --aspect_attn=false --auto_encoder=false --mode="adversarial_train"

cd ..
mkdir DPGAN_baseline_multi_doc_LDA_topic5_4
mv DPGAN_copy/myexperiment DPGAN_baseline_multi_doc_LDA_topic5_4
mv DPGAN_copy/test_max_generated DPGAN_baseline_multi_doc_LDA_topic5_4
mv DPGAN_copy/test_sample_generated DPGAN_baseline_multi_doc_LDA_topic5_4
mv DPGAN_copy/train_sample_generated DPGAN_baseline_multi_doc_LDA_topic5_4
mv DPGAN_copy/pretrain_test_sample_generated DPGAN_baseline_multi_doc_LDA_topic5_4
mv DPGAN_copy/MLE DPGAN_baseline_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_test DPGAN_baseline_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_train DPGAN_baseline_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_result DPGAN_baseline_multi_doc_LDA_topic5_4
rmdir -rf DPGAN_copy/__pycache__

# aspect (No AE)
cd DPGAN_copy
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --auto_encoder=false --mode="train_generator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --auto_encoder=false --mode="train_discriminator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --auto_encoder=false --mode="adversarial_train"

cd ..
mkdir DPGAN_aspect_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/myexperiment DPGAN_aspect_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/test_max_generated DPGAN_aspect_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/test_sample_generated DPGAN_aspect_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/train_sample_generated DPGAN_aspect_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/pretrain_test_sample_generated DPGAN_aspect_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/MLE DPGAN_aspect_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_test DPGAN_aspect_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_train DPGAN_aspect_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_result DPGAN_aspect_only_multi_doc_LDA_topic5_4
rmdir -rf DPGAN_copy/__pycache__

# senti (No AE)
cd DPGAN_copy
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --aspect_attn=false --auto_encoder=false --mode="train_generator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --aspect_attn=false --auto_encoder=false --mode="train_discriminator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --aspect_attn=false --auto_encoder=false --mode="adversarial_train"

cd ..
mkdir DPGAN_senti_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/myexperiment DPGAN_senti_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/test_max_generated DPGAN_senti_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/test_sample_generated DPGAN_senti_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/train_sample_generated DPGAN_senti_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/pretrain_test_sample_generated DPGAN_senti_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/MLE DPGAN_senti_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_test DPGAN_senti_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_train DPGAN_senti_only_multi_doc_LDA_topic5_4
mv DPGAN_copy/discriminator_result DPGAN_senti_only_multi_doc_LDA_topic5_4
rmdir -rf DPGAN_copy/__pycache__