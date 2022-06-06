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
                nlps[language] = stanza.Pipeline('en', tokenize_no_ssplit=True)
            elif language == 'German':
                nlps[language] = stanza.Pipeline('de', tokenize_no_ssplit=True)
            elif language == 'Japanese':
                nlps[language] = stanza.Pipeline('de', tokenize_no_ssplit=True)
            elif language == 'Chinese':
                nlps[language] = stanza.Pipeline('zh', tokenize_no_ssplit=True)
            elif language == 'French':
                nlps[language] = stanza.Pipeline('ja', tokenize_no_ssplit=True)

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
    def is_root_be_verb(analyzed_sentence, language):
        """ Check if the root is the be verb, assuming that there's a single root. """
        if language == 'English':
            be_verb = 'be'
        elif language == 'German':
            be_verb = 'sein'
        elif language == 'Chinese':
            be_verb = '有'
        elif language == 'Japanese':
            be_verb = 'ある'

        root = [token for token in analyzed_sentence if token['dep'].lower() == 'root'][0]
        return root['lemma'].lower() == be_verb

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
        agg_token_lists = [[x.to_dict() for x in y.tokens] for y in analyzed_sentence.sentences]
        token_lists = []
        for agg_token_list in agg_token_lists:
            token_lists.append([])
            for agg_token in agg_token_list:
                if len(agg_token) == 1:
                    token_lists[-1].append(agg_token[0])
                else:
                    if language != 'German':
                        print('Tokens longer than 1 in sentence: ' + analyzed_sentence.text)
                        assert False
                    token_lists[-1] += agg_token[1:]
        return [[{
            'pos': x['upos'],
            'dep': x['deprel'],
            'lemma': x['lemma'],
            'head_ind': x['head'] - 1
        } for x in y] for y in token_lists]
