import sys

from tf_idf.tf_idf import TFIDFJob


def concatenate_words(query):
    return " and ".join(word.upper() for word in query)


def main():
    query = set(word.lower() for word in sys.argv[1:])
    if len(query) == 0:
        sys.exit('Error: no search query passed')

    result_dict = {}

    job = TFIDFJob(args=['input/tf_idf/*'])
    with job.make_runner() as runner:
        runner.run()
        for key, tfidf in job.parse_output(runner.cat_output()):
            word = key[0]
            doc = key[1]
            if word in query:
                if doc not in result_dict:
                    result_dict[doc] = []
                result_dict[doc].append(tfidf)

    if len(result_dict) == 0:
        sys.exit(f'Error: found no documents containing words {concatenate_words(query)}')

    result_dict = {doc: sum(result_dict[doc]) / len(result_dict[doc]) for doc in result_dict}
    doc = max(result_dict, key=result_dict.get)

    sys.exit(f'{doc} - most relevant document containing words {concatenate_words(query)}')


if __name__ == '__main__':
    main()
