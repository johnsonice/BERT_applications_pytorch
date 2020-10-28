export TRAIN_FILE=../../data/wikizh/wiki_zh_train.txt
export LTP_RESOURCE=../../data/pre_trained_model/ltp/base
export BERT_RESOURCE=../../data/pre_trained_model/chinese-bert-wwm
export SAVE_PATH=../../data/wikizh/ref.txt

python ../chinese_ref.py \
    --file_name=$TRAIN_FILE \
    --ltp=$LTP_RESOURCE	\
    --bert=$BERT_RESOURCE \
    --save_path=$SAVE_PATH 
