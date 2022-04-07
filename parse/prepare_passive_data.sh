#!/bin/bash
if [ "$5" == "translated" ]; then
	translated_arg="--translated"
	translated_str="_translated"
else
	translated_arg=""
	translated_str=""
fi

bash parse.sh $1 $2 $3 $4 $5

cd tmv-annotator-tool

parsed_file_path=../../cached_dataset_files/$3${translated_str}_$2_parsed.txt
annotated_file_path=tmv_out_$2_$3${translated_str}

if [ "$2" == "English" ]; then
	$1 TMV-EN.py $parsed_file_path $annotated_file_path
fi

if [ "$2" == "German" ]; then
	$1 TMV-DE.py $parsed_file_path $annotated_file_path seinVerbs.txt
fi

if [ "$2" == "French" ]; then
	$1 TMV-FR.py $parsed_file_path $annotated_file_path etreVerbs.txt
fi

cd ../..

cp parse/tmv-annotator-tool/output/$annotated_file_path.verbs cached_dataset_files
