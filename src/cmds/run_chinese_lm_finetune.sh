#export CUDA_VISIBLE_DEVICES=0
export TRAIN_FILE=../../data/wikizh/wiki_zh_train.txt
export TEST_FILE=../../data/wikizh/wiki_zh_test.txt
export REF_FILE=../../data/wikizh/ref.txt
export BERT_RESOURCE=../../data/trained_model/chinese/checkpoint-71000 #../../data/pre_trained_model/chinese-bert-wwm
export TOKENIZER_RESOURCE=../../data/pre_trained_model/chinese-bert-wwm
export OUTPUT=../../data/trained_model/chinese
#export RUN_NAME=../../data/trained_model/chinese/run

python ../run_language_modeling.py \
    --output_dir=$OUTPUT \
    --model_type=bert \
    --model_name_or_path=$BERT_RESOURCE \
    --tokenizer_name=$TOKENIZER_RESOURCE \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --chinese_ref_file=$REF_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --line_by_line \
    --save_total_limit=5 \
    --whole_word_mask \
    --overwrite_output_dir \
    --logging_steps=500 \
    --eval_steps=500 \
    --save_steps=500 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
