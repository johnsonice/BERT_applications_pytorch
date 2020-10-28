#export CUDA_VISIBLE_DEVICES=0
export TRAIN_FILE=../../data/wikitext-2/wiki.train.tokens
export TEST_FILE=../../data/wikitext-2/wiki.test.tokens
export OUTPUT=../../data/trained_model/english

python ../run_language_modeling.py \
    --output_dir=$OUTPUT \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --whole_word_mask
