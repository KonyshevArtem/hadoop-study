import os
import math

from mrjob.job import MRJob
from mrjob.step import MRStep

DOC_COUNT = 0


def get_file_name():
    return os.path.basename(os.environ['mapreduce_map_input_file'])


class DocWordFrequencyStep(MRStep):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, _, line):
        doc = get_file_name()
        for word in line.split():
            yield (doc, word), 1

    def reducer(self, doc_word, amounts_generator):
        yield doc_word, sum(amounts_generator)


class DocWordCountStep(MRStep):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, key, amount):
        doc = key[0]
        word = key[1]
        yield doc, (word, amount)

    def reducer(self, doc, values_generator):
        values_list = list(values_generator)
        doc_words_count = sum(word_count[1] for word_count in values_list)
        for values in values_list:
            word = values[0]
            word_count = values[1]
            yield (word, doc), word_count / doc_words_count


class CorpusWordFrequencyStep(MRStep):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, key, tf):
        word = key[0]
        doc = key[1]
        yield word, (doc, tf, 1)

    def reducer(self, word, values_generator):
        values_list = list(values_generator)
        corpus_word_count = sum(values[2] for values in values_list)
        for values in values_list:
            doc = values[0]
            tf = values[1]
            yield (word, doc), (tf, corpus_word_count)


class TFIDFStep(MRStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, word_doc, values):
        global DOC_COUNT
        tf = values[0]
        corpus_word_count = values[1]
        idf = math.log(DOC_COUNT / corpus_word_count)
        yield word_doc, tf * idf

    def reducer(self, word_doc, tfidf_generator):
        for tfidf in tfidf_generator:
            yield word_doc, tfidf


class TFIDFJob(MRJob):

    def __init__(self, args=None):
        self.doc_count = 0
        super().__init__(args)

    def run(self):
        global DOC_COUNT
        DOC_COUNT = len(self.options.args)
        super().run()

    def get_doc_count(self):
        return self.doc_count

    def steps(self):
        return [
            DocWordFrequencyStep(),
            DocWordCountStep(),
            CorpusWordFrequencyStep(),
            TFIDFStep()
        ]


if __name__ == '__main__':
    TFIDFJob().run()
