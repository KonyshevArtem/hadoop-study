import os

from mrjob.job import MRJob
from mrjob.step import MRStep


def get_file_name():
    return os.path.basename(os.environ['mapreduce_map_input_file'])


class DocWordFrequencyStep(MRStep):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, _, line):
        for word in line.split():
            yield (get_file_name(), word), 1

    def reducer(self, doc_word, amounts):
        yield doc_word, sum(amounts)


class DocWordCountStep(MRStep):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, mapper=self.mapper, reducer=self.reducer)

    def mapper(self, doc_word, amount):
        yield doc_word[0], (doc_word[1], amount)

    def reducer(self, doc, words_counts):
        words_counts = list(words_counts)
        doc_words_count = sum(word_count[1] for word_count in words_counts)
        for word_count in words_counts:
            yield (word_count[0], doc), (word_count[1], doc_words_count)


class TFIDF(MRJob):
    def steps(self):
        return [
            DocWordFrequencyStep(),
            DocWordCountStep()
        ]


if __name__ == '__main__':
    TFIDF().run()
