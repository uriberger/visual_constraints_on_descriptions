from spacy import load

""" This file contains functions and definitions related to text used across all the project. """

nlp = load('en_core_web_sm')


""" Given a sentence analyze by spaCy, check if its main verb is transitive. """


def is_transitive_sentence(analyzed_sentence):
    return len([token for token in analyzed_sentence if token.dep_ == 'dobj' and token.head.dep_ == 'ROOT']) > 0
