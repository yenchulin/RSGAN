:: Please first move this file outside RSGAN
:: and copy RSGAN folder without any result naming RSGAN_copy


:: epoch * 5/8
:: both (No AE)
cd RSGAN_copy
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --auto_encoder=false --mode="train_generator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --auto_encoder=false --mode="train_discriminator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --auto_encoder=false --mode="adversarial_train"

cd ..
mkdir RSGAN_both_multi_doc_LDA_topic5_3
move RSGAN_copy\myexperiment RSGAN_both_multi_doc_LDA_topic5_3
move RSGAN_copy\test_max_generated RSGAN_both_multi_doc_LDA_topic5_3
move RSGAN_copy\test_sample_generated RSGAN_both_multi_doc_LDA_topic5_3
move RSGAN_copy\train_sample_generated RSGAN_both_multi_doc_LDA_topic5_3
move RSGAN_copy\pretrain_test_sample_generated RSGAN_both_multi_doc_LDA_topic5_3
move RSGAN_copy\MLE RSGAN_both_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_test RSGAN_both_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_train RSGAN_both_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_result RSGAN_both_multi_doc_LDA_topic5_3
rmdir /s /q RSGAN_copy\__pycache__

:: baseline (No AE)
cd RSGAN_copy
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --aspect_attn=false --auto_encoder=false --mode="train_generator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --aspect_attn=false --auto_encoder=false --mode="train_discriminator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --aspect_attn=false --auto_encoder=false --mode="adversarial_train"

cd ..
mkdir RSGAN_baseline_multi_doc_LDA_topic5_3
move RSGAN_copy\myexperiment RSGAN_baseline_multi_doc_LDA_topic5_3
move RSGAN_copy\test_max_generated RSGAN_baseline_multi_doc_LDA_topic5_3
move RSGAN_copy\test_sample_generated RSGAN_baseline_multi_doc_LDA_topic5_3
move RSGAN_copy\train_sample_generated RSGAN_baseline_multi_doc_LDA_topic5_3
move RSGAN_copy\pretrain_test_sample_generated RSGAN_baseline_multi_doc_LDA_topic5_3
move RSGAN_copy\MLE RSGAN_baseline_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_test RSGAN_baseline_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_train RSGAN_baseline_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_result RSGAN_baseline_multi_doc_LDA_topic5_3
rmdir /s /q RSGAN_copy\__pycache__

:: aspect (No AE)
cd RSGAN_copy
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --auto_encoder=false --mode="train_generator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --auto_encoder=false --mode="train_discriminator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --senti_attn=false --auto_encoder=false --mode="adversarial_train"

cd ..
mkdir RSGAN_aspect_only_multi_doc_LDA_topic5_3
move RSGAN_copy\myexperiment RSGAN_aspect_only_multi_doc_LDA_topic5_3
move RSGAN_copy\test_max_generated RSGAN_aspect_only_multi_doc_LDA_topic5_3
move RSGAN_copy\test_sample_generated RSGAN_aspect_only_multi_doc_LDA_topic5_3
move RSGAN_copy\train_sample_generated RSGAN_aspect_only_multi_doc_LDA_topic5_3
move RSGAN_copy\pretrain_test_sample_generated RSGAN_aspect_only_multi_doc_LDA_topic5_3
move RSGAN_copy\MLE RSGAN_aspect_only_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_test RSGAN_aspect_only_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_train RSGAN_aspect_only_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_result RSGAN_aspect_only_multi_doc_LDA_topic5_3
rmdir /s /q RSGAN_copy\__pycache__

:: senti (No AE)
cd RSGAN_copy
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --aspect_attn=false --auto_encoder=false --mode="train_generator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --aspect_attn=false --auto_encoder=false --mode="train_discriminator"
python main.py --adver_epoch=25 --g_pre_epoch=100 --d_pre_epoch=150 --aspect_attn=false --auto_encoder=false --mode="adversarial_train"

cd ..
mkdir RSGAN_senti_only_multi_doc_LDA_topic5_3
move RSGAN_copy\myexperiment RSGAN_senti_only_multi_doc_LDA_topic5_3
move RSGAN_copy\test_max_generated RSGAN_senti_only_multi_doc_LDA_topic5_3
move RSGAN_copy\test_sample_generated RSGAN_senti_only_multi_doc_LDA_topic5_3
move RSGAN_copy\train_sample_generated RSGAN_senti_only_multi_doc_LDA_topic5_3
move RSGAN_copy\pretrain_test_sample_generated RSGAN_senti_only_multi_doc_LDA_topic5_3
move RSGAN_copy\MLE RSGAN_senti_only_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_test RSGAN_senti_only_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_train RSGAN_senti_only_multi_doc_LDA_topic5_3
move RSGAN_copy\discriminator_result RSGAN_senti_only_multi_doc_LDA_topic5_3
rmdir /s /q RSGAN_copy\__pycache__