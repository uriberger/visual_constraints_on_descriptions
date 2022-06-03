from spacy import load
import stanza


nlps = {}
tokenizers = {}


class TextUtils:
    """ This class contains functions and definitions related to text used across all the project. """

    @staticmethod
    def get_nlp(language):
        global nlps
        if language not in nlps:
            if language == 'English':
                nlps[language] = load('en_core_web_sm')
            elif language == 'German':
                nlps[language] = load('de_core_news_sm')
            elif language == 'Japanese':
                nlps[language] = load('ja_core_news_sm')
            elif language == 'Chinese':
                #nlps[language] = load('zh_core_web_sm')
                nlps[language] = stanza.Pipeline('zh')
            elif language == 'French':
                nlps[language] = load('fr_core_news_sm')

        return nlps[language]

    @staticmethod
    def get_tokenizer(language):
        return TextUtils.get_nlp(language).tokenizer

    """ Given a sentence analyze by spaCy, check if its main verb is transitive. This is done by searching if there's a
        token which is a direct object of the root.
    """

    @staticmethod
    def is_transitive_sentence(analyzed_sentence, language):
        if language == 'English':
            direct_object_dep_tag = 'dobj'
        elif language == 'German':
            direct_object_dep_tag = 'oa'
        elif language == 'Chinese':
            # direct_object_dep_tag = 'dobj'
            direct_object_dep_tag = 'obj'
        elif language == 'Japanese':
            direct_object_dep_tag = 'obj'
        elif language == 'French':
            direct_object_dep_tag = 'obj'
        return len([token for token in analyzed_sentence
                    if token['dep'] == direct_object_dep_tag and
                    analyzed_sentence[token['head_ind']]['dep'].lower() == 'root']) > 0

    @staticmethod
    def tokenize(sentence, language):
        """ Tokenize an input sentence. """
        return [str(x) for x in list(TextUtils.get_tokenizer(language)(sentence.lower()))]

    @staticmethod
    def preprocess_token(token):
        token = "".join(c for c in token if c not in ("?", ".", ";", ":", "!", "'", "`", "\""))
        token = token.lower()

        return token

    @staticmethod
    def tokenize_and_clean(sentence, language):
        tokenized_sentence = TextUtils.tokenize(sentence, language)
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

    @staticmethod
    def extract_nlp_info(language, analyzed_sentence):
        if language == 'Chinese':
            # In Chinese we use Stanza
            token_lists = [[x.to_dict() for x in y.tokens] for y in analyzed_sentence.sentences]
            if max([max([len(y) for y in x]) for x in token_lists]) > 1:
                print('Tokens longer than 1 in sentence: ' + analyzed_sentence.text)
                assert False
            token_lists = [[x[0] for x in y] for y in token_lists]
            return [[{
                'start': x['start_char'],
                'pos': x['xpos'],
                'dep': x['deprel'],
                'lemma': x['lemma'],
                'head_ind': x['head'] - 1
            } for x in y] for y in token_lists]
        else:
            # In other language we use spaCy
            return [{
                'start': x.idx,
                'pos': x.pos_,
                'dep': x.dep_,
                'lemma': x.lemma_,
                'head_ind': x.head.i
            } for x in analyzed_sentence]
