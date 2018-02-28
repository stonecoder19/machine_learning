
def wc_mapper(document):
    for word in tokenize(document):
        yield (word, 1)

def wc_reducer(word, counts):
    yield (word, sum(counts))


def word_count(documents):

    collector = defaultdict(list)

    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)

    return [output
            for word, counts in collector.iteritems()
            for output in wc_redcuer(word,counts)]
