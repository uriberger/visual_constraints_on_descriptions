from spacy import load
from spacy.matcher import Matcher, DependencyMatcher


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

    """ Given a sentence analyze by spaCy, check if its main verb is transitive. """

    @staticmethod
    def is_transitive_sentence(analyzed_sentence):
        return len([token for token in analyzed_sentence if token.dep_ == 'dobj' and token.head.dep_ == 'ROOT']) > 0

    """ Get the spaCy pattern matcher for finding passive sentences.
        Each language's dependency trees are different, so the matcher depends on the language.
    """

    @staticmethod
    def get_passive_matcher():
        vocab = TextUtils.get_nlp().vocab
        if TextUtils.get_language() == 'English':
            matcher = Matcher(vocab)
            pattern = [
                {'DEP': 'nsubjpass'},
                {'DEP': 'aux', 'OP': '*'},
                {'DEP': 'auxpass'},
                {'TAG': 'VBN'}
            ]
        elif TextUtils.get_language() == 'German':
            ''' In German, a sentence should be classified as passive if:
            1. There's a token with the 'oc' dependency tag (clausal object), and
            2. The POS tag of this token is a verb, and
            3. The POS tag of this token's parent in the dependency tree is AUX

            E.g., in the sentence "Der Apfel wird gegessen" ("The apple became eaten", the passive verb is "gegessen",
            its dependency tag is 'oc', and its head is the "wird" token, with the POS tag AUX.
            '''
            matcher = DependencyMatcher(vocab)
            pattern = [
                {
                    'RIGHT_ID': 'auxiliary_verb',
                    'RIGHT_ATTRS': {'POS': 'AUX', 'LEMMA': 'werden'}
                },
                {
                    'LEFT_ID': 'auxiliary_verb',
                    'REL_OP': '>',
                    'RIGHT_ID': 'passive_verb',
                    'RIGHT_ATTRS': {'DEP': 'oc', 'POS': 'VERB'}
                }
            ]
        else:
            matcher = Matcher(vocab)
            pattern = []
        matcher.add('Passive', [pattern])
        return matcher

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
