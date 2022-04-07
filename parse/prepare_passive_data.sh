#!/bin/bash
if [ "$5" == "translated" ]; then
	translated_arg="--translated"
	translated_str="_translated"
else
	translated_arg=""
	translated_str=""
fi

bash parse.sh $1 $2 $3 $4 $5

cur_dir=$PWD
py_path=$cur_dir/$1

cd tmv-annotator-tool

parsed_train_file_path=../../cached_dataset_files/$3${translated_str}_$2_train_parsed.txt
parsed_val_file_path=../../cached_dataset_files/$3${translated_str}_$2_val_parsed.txt
annotated_train_file_path=tmv_out_$2_$3${translated_str}_train
annotated_val_file_path=tmv_out_$2_$3${translated_str}_val

if [ "$2" == "English" ]; then
	$py_path TMV-EN.py $parsed_train_file_path $annotated_train_file_path
	$py_path TMV-EN.py $parsed_val_file_path $annotated_val_file_path
fi

if [ "$2" == "German" ]; then
	$py_path TMV-DE.py $parsed_train_file_path $annotated_train_file_path seinVerbs.txt
	$py_path TMV-DE.py $parsed_val_file_path $annotated_val_file_path seinVerbs.txt
fi

if [ "$2" == "French" ]; then
	$py_path TMV-FR.py $parsed_train_file_path $annotated_train_file_path etreVerbs.txt
	$py_path TMV-FR.py $parsed_val_file_path $annotated_val_file_path etreVerbs.txt
fi

cd ../..

cp parse/tmv-annotator-tool/output/$annotated_train_file_path.verbs cached_dataset_files
cp parse/tmv-annotator-tool/output/$annotated_val_file_path.verbs cached_dataset_files
