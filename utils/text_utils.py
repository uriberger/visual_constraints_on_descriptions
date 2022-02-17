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

        return nlps[language]

    @staticmethod
    def get_tokenizer():
        return TextUtils.get_nlp().tokenizer

    """ Given a sentence analyze by spaCy, check if its main verb is transitive. """

    @staticmethod
    def is_transitive_sentence(analyzed_sentence):
        return len([token for token in analyzed_sentence if token.dep_ == 'dobj' and token.head.dep_ == 'ROOT']) > 0

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
