from spacy import load


used_language = 'English'
nlps = {}
tokenizers = {}


class TextUtils:
    """ This class contains functions and definitions related to text used across all the project. """

    @staticmethod
    def set_language(language):
        global used_language
        used_language = language

    @staticmethod
    def get_language():
        return used_language

    @staticmethod
    def get_nlp():
        global nlps
        language = TextUtils.get_language()
        if language not in nlps:
            if language == 'English':
                nlps[language] = load('en_core_web_sm')
            elif language == 'German':
                nlps[language] = load('de_core_news_sm')
            elif language == 'Japanese':
                nlps[language] = load('ja_core_news_sm')
            elif language == 'Chinese':
                nlps[language] = load('zh_core_web_sm')
            elif language == 'French':
                nlps[language] = load('fr_core_news_sm')

        return nlps[language]

    @staticmethod
    def get_tokenizer():
        return TextUtils.get_nlp().tokenizer

    """ Given a sentence analyze by spaCy, check if its main verb is transitive. This is done by searching if there's a
        token which is a direct object of the root.
    """

    @staticmethod
    def is_transitive_sentence(analyzed_sentence):
        language = TextUtils.get_language()
        if language == 'English':
            direct_object_dep_tag = 'dobj'
        elif language == 'German':
            direct_object_dep_tag = 'oa'
        elif language == 'Chinese':
            direct_object_dep_tag = 'dobj'
        elif language == 'Japanese':
            direct_object_dep_tag = 'obj'
        elif language == 'French':
            # Not implemented yet
            direct_object_dep_tag = 'NONE'
        return len([token for token in analyzed_sentence
                    if token.dep_ == direct_object_dep_tag and token.head.dep_ == 'ROOT']) > 0

    @staticmethod
    def tokenize(sentence):
        """ Tokenize an input sentence. """
        return [str(x) for x in list(TextUtils.get_tokenizer()(sentence.lower()))]

    @staticmethod
    def preprocess_token(token):
        token = "".join(c for c in token if c not in ("?", ".", ";", ":", "!", "'", "`", "\""))
        token = token.lower()

        return token

    @staticmethod
    def tokenize_and_clean(sentence):
        tokenized_sentence = TextUtils.tokenize(sentence)
        return [TextUtils.preprocess_token(x) for x in tokenized_sentence]

    @staticmethod
    def phrase_in_sent(tokenized_sentence, tokenized_phrase):
        for sent_ext_ind in range(len(tokenized_sentence)):
            phrase_ind = 0
            sent_int_ind = sent_ext_ind
            while phrase_ind < len(tokenized_phrase) and sent_int_ind < len(tokenized_sentence) and\
                    tokenized_sentence[sent_int_ind] == tokenized_phrase[phrase_ind]:
                phrase_ind += 1
                sent_int_ind += 1
            if phrase_ind == len(tokenized_phrase):
                return True
        return False
