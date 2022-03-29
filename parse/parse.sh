#!/bin/bash
cd ..

if [ "$4" == "translated" ]; then
	translated_arg="--translated"
	translated_str="_translated"
else
	translated_arg=""
	translated_str=""
fi

$1 main.py --language $2 --dataset $3 --dump_captions $translated_arg
cd parse/mate_parser
javac LemmatizeAndParse.java
java LemmatizeAndParse $2 $3 $translated_arg

parsed_file_name=$3${translated_str}_$2_parsed.txt

mv $parsed_file_name ../../cached_dataset_files