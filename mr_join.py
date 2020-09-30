from mrjob.job import MRJob
from mrjob.step import MRStep
import os
import itertools


def get_file_name():
    return os.path.basename(os.environ['mapreduce_map_input_file'])


class MRJoin(MRJob):

    def mapper(self, _, row):
        attributes = row.split(',')
        yield attributes[0], (get_file_name(), attributes[1:])

    def reducer(self, key, value):
        d = {}
        for row in value:
            if row[0] not in d:
                d[row[0]] = []
            d[row[0]].append(row[1])
        yield key, list(d.values())

    def joiner(self, key, value):
        for row in value:
            for c in itertools.product(*row):
                yield key, list(itertools.chain(*c))

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer),
            MRStep(reducer=self.joiner)
        ]


if __name__ == '__main__':
    MRJoin.run()
