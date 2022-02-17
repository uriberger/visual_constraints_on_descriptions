from spacy import load

""" This file contains functions and definitions related to text used across all the project. """

nlp = load('en_core_web_sm')
tokenizer = nlp.tokenizer


""" Given a sentence analyze by spaCy, check if its main verb is transitive. """


def is_transitive_sentence(analyzed_sentence):
    return len([token for token in analyzed_sentence if token.dep_ == 'dobj' and token.head.dep_ == 'ROOT']) > 0


def tokenize(sentence):
    """ Tokenize and clean an input sentence. """
    return [str(x) for x in list(tokenizer(sentence.lower()))]


def preprocess_token(token):
    token = "".join(c for c in token if c not in ("?", ".", ";", ":", "!", "'", "`", "\""))
    token = token.lower()

    return token


def tokenize_and_clean(sentence):
    tokenized_sentence = tokenize(sentence)
    return [preprocess_token(x) for x in tokenized_sentence]
