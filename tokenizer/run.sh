db_root= #SET YOUR DB PATH HERE
mkdir -p $db_root/sets
cd ../preprocessing/splitting/
python train_test_split.py --infile $db_root/mcv/complete.json --save_dir $db_root/sets --prefix "mcv-" --unique
python train_test_split.py --infile $db_root/swc/complete.json --save_dir $db_root/sets --prefix "swc-" --unique
python train_test_split.py --infile $db_root/mai/complete.json --save_dir $db_root/sets --prefix "mai-" --unique
cd ../../tokenizer
python train.py hparams/unigram_5000.yaml --data_folder $db_root/sets
