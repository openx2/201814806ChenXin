import heapq
from collections import Counter
from collections import defaultdict

class KNNClassifier(object):
    def __init__(self, vsms, labels):
        self.vsms = vsms
        self.labels = labels

    def buildInvertedList(self):
        self.invertedList = defaultdict(list)
        for i, vsm in enumerate(self.vsms):
            for t in vsm.getTerms():
                self.invertedList[t].append(i)

    def train(self):
        self.buildInvertedList()

    def classify(self, vsm, k):
        priority_queue = []
        index_set = set() # avoid duplicate indices
        for t in vsm.getTerms():
            for index in self.invertedList[t]:
                if index not in index_set:
                    cos_similarity = vsm.dot(self.vsms[index])
                    priority_queue.append((cos_similarity, index))
                    index_set.add(index)
        k_nearest_neighbor = heapq.nlargest(k, priority_queue)
        label_counter = Counter(self.labels[i] for dis, i in k_nearest_neighbor)
        #debug
        #return k_nearest_neighbor
        return label_counter.most_common(1)[0][0]
