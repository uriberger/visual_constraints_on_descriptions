#!/bin/bash
cur_dir=$PWD
cd ..

if [ "$5" == "translated" ]; then
	translated_arg="--translated"
	translated_str="_translated"
else
	translated_arg=""
	translated_str=""
fi

datasets_dir_path=$cur_dir/$4

$cur_dir/$1 main.py --language $2 --dataset $3 --datasets_dir $datasets_dir_path --dump_captions $translated_arg
cd $cur_dir/mate_parser
javac LemmatizeAndParse.java
java LemmatizeAndParse $2 $3 $translated_arg

parsed_train_file_name=$3${translated_str}_$2_train_parsed.txt
parsed_val_file_name=$3${translated_str}_$2_val_parsed.txt

mv $parsed_train_file_name ../../cached_dataset_files
mv $parsed_val_file_name ../../cached_dataset_files
