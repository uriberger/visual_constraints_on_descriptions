#!/bin/bash
cd ..
$1 main.py --language $2 --dataset $3 --dump_captions
cd english_german_french_passive/mate_parser
javac LemmatizeAndParse.java
java LemmatizeAndParse $2 $3

cd ../tmv-annotator-tool

if [ "$2" == "English" ]; then
	$1 TMV-EN.py ../mate_parser/parsed.txt tmv_out_$2_$3
fi

if [ "$2" == "German" ]; then
	$1 TMV-DE.py ../mate_parser/parsed.txt tmv_out_$2_$3 seinVerbs.txt
fi

if [ "$2" == "French" ]; then
	$1 TMV-FR.py ../mate_parser/parsed.txt tmv_out_$2_$3 etreVerbs.txt
fi

cd ../..

cp english_german_french_passive/tmv-annotator-tool/output/tmv_out_$2_$3.verbs cached_dataset_files