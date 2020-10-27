import math
import os
import re

from mrjob.job import MRJob
from mrjob.step import MRStep

WORD_DELIMITERS = r'[ ,.()?!:;\t\n\[\]=\\\/]'
WORD_BLACKLIST = ['a', 'the', 'an', 'is']


def get_file_name():
    return os.path.basename(os.environ['mapreduce_map_input_file'])


class DocCountStep(MRStep):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, _, line):
        doc = get_file_name()
        yield 1, (doc, line)

    def reducer(self, _, values_generator):
        values_list = list(values_generator)
        doc_count = len(set(values[0] for values in values_list))
        for values in values_list:
            doc = values[0]
            line = values[1]
            yield (doc, doc_count), line


class DocWordFrequencyStep(MRStep):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, key, line):
        doc = key[0]
        doc_count = key[1]
        for word in re.split(WORD_DELIMITERS, line):
            if len(word) > 0 and word.lower() not in WORD_BLACKLIST:
                yield (doc, word, doc_count), 1

    def reducer(self, key, amounts_generator):
        yield key, sum(amounts_generator)


class DocWordCountStep(MRStep):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, key, amount):
        doc = key[0]
        word = key[1]
        doc_count = key[2]
        yield (doc, doc_count), (word, amount)

    def reducer(self, key, values_generator):
        doc = key[0]
        doc_count = key[1]
        values_list = list(values_generator)
        doc_words_count = sum(word_count[1] for word_count in values_list)
        for values in values_list:
            word = values[0]
            word_count = values[1]
            yield (word, doc, doc_count), word_count / doc_words_count


class CorpusWordFrequencyStep(MRStep):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, key, tf):
        word = key[0]
        doc = key[1]
        doc_count = key[2]
        yield (word, doc_count), (doc, tf, 1)

    def reducer(self, key, values_generator):
        word = key[0]
        doc_count = key[1]
        values_list = list(values_generator)
        corpus_word_count = sum(values[2] for values in values_list)
        for values in values_list:
            doc = values[0]
            tf = values[1]
            yield (word, doc, doc_count), (tf, corpus_word_count)


class TFIDFStep(MRStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, key, values):
        word = key[0]
        doc = key[1]
        doc_count = key[2]
        tf = values[0]
        corpus_word_count = values[1]
        idf = math.log(doc_count / corpus_word_count)
        yield (word, doc), tf * idf

    def reducer(self, word_doc, tfidf_generator):
        for tfidf in tfidf_generator:
            yield word_doc, tfidf


class TFIDFJob(MRJob):

    def __init__(self, args=None):
        super().__init__(args)

    def steps(self):
        return [
            DocCountStep(),
            DocWordFrequencyStep(),
            DocWordCountStep(),
            CorpusWordFrequencyStep(),
            TFIDFStep()
        ]


if __name__ == '__main__':
    TFIDFJob().run()
