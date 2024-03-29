# English, German and French parsing

To parse data using the mate parser, do the following steps:

1. Download the Mate-partser transition jar file (this is the download page: https://code.google.com/archive/p/mate-tools/wikis/ParserAndModels.wiki, this is the url of the download itself: https://drive.google.com/file/d/0B-qbj-8rtoUMbVEzWDlvd0ZxVFU/edit?usp=sharing)
2. Extract the jar files and copy the directories (examples, is2, META-INF, org) into the mate_parser directory
3. Create a directory named 'models' under the mate_parser directory
4. Download the following models and put all in the 'models' directory you just created:
	- French lemmatizer: File name 'lemma-fra.mdl', download page: https://code.google.com/archive/p/mate-tools/wikis/ParserAndModels.wiki, url of download https://drive.google.com/file/d/0B-qbj-8rtoUMMEYwY0FFLUVmeEU/edit?resourcekey=0-5wexOEy2UbQzglwCncMsSg
	- German lemmatizer: File name 'lemma-ger-3.6.model', download page: https://code.google.com/archive/p/mate-tools/wikis/ParserAndModels.wiki, url of download https://drive.google.com/file/d/0B-qbj-8rtoUMaUVsWUFuOE81ZW8/edit?resourcekey=0-tujCzsTsvBfwcNoMa2wMUQ
	- English lemmatizer: Need to download the small-models-english-tgz file (download page https://code.google.com/archive/p/mate-tools/downloads?page=1, url of download https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/mate-tools/small-models-english.tgz)
	- French parser: File name 'pet-fra-S2apply-40-0.25-0.1-2-2-ht4-hm4-kk0', download page: https://code.google.com/archive/p/mate-tools/wikis/ParserAndModels.wiki, url of download https://drive.google.com/file/d/0B-qbj-8rtoUMYV8yalAtcFk0WFE/edit?resourcekey=0-iNvh3H5Esglkn_0NJbqDcQ
	- German parser: File name 'pet-ger-S2a-40-0.25-0.1-2-2-ht4-hm4-kk0', download page: https://code.google.com/archive/p/mate-tools/wikis/ParserAndModels.wiki, url of download https://drive.google.com/file/d/0B-qbj-8rtoUMLUg5NGpBVW9JNkE/edit?usp=sharing
	- English parser: File name 'per-eng-S2b-40.mdl', download page: https://code.google.com/archive/p/mate-tools/wikis/ParserAndModels.wiki, url of download https://drive.google.com/file/d/0B-qbj-8rtoUMWWlINVhqdTU0bjQ/view?resourcekey=0-ZIdlt4QD3NA-NHyKFMJeqw
5. Run the 'parse.sh' script with the following arguments: 
	parse.sh <relative path to python exectuble> <Language (English/German/French)> <dataset name> <relative path to datasets dir>
Or
	parse.sh <relative path to python exectuble> <Language (English/German/French)> <dataset name> <relative path to datasets dir> translated
If the captions in the dataset are translated.	

To prepare the passive analysis data, run the 'preparse_passive_data.sh' script with the following arguments: 
	prepare_passive_data.sh <relative path to python exectuble> <Language (English/German/French)> <dataset name> <relative path to datasets dir>
Or
	prepare_passive_data.sh <relative path to python exectuble> <Language (English/German/French)> <dataset name> <relative path to datasets dir> translated
If the captions in the dataset are translated.	
