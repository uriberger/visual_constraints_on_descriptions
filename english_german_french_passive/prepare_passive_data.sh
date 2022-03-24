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
cd english_german_french_passive/mate_parser
javac LemmatizeAndParse.java
java LemmatizeAndParse $2 $3 $translated_arg

cd ../tmv-annotator-tool

if [ "$2" == "English" ]; then
	$1 TMV-EN.py ../mate_parser/parsed.txt tmv_out_$2_$3$translated_str
fi

if [ "$2" == "German" ]; then
	$1 TMV-DE.py ../mate_parser/parsed.txt tmv_out_$2_$3$translated_str seinVerbs.txt
fi

if [ "$2" == "French" ]; then
	$1 TMV-FR.py ../mate_parser/parsed.txt tmv_out_$2_$3$translated_str etreVerbs.txt
fi

cd ../..

cp english_german_french_passive/tmv-annotator-tool/output/tmv_out_$2_$3$translated_str.verbs cached_dataset_files