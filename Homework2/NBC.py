import math
from collections import Counter
from collections import defaultdict

class NaiveBayesClassifier(object):
    def __init__(self, tokenized_docs, labels):
        self.training_data = tokenized_docs
        self.labels = labels

    def train(self):
        class_counter = defaultdict(Counter)
        vocabulary_length = 0
        for i, doc in enumerate(self.training_data):
            class_counter[self.labels[i]].update(doc)
            vocabulary_length += len(class_counter[self.labels[i]])

        label_word_num = {lb:sum(c.values()) for lb, c in class_counter.items()}
        self.label_missing_word_prob = {lb:math.log(1/(label_word_num[lb] + vocabulary_length)) for lb in class_counter}

        self.label_word_probability = {lb:dict() for lb in class_counter}
        for lb in class_counter:
            for w in class_counter[lb]:
                prob = (class_counter[lb][w] + 1)/(label_word_num[lb] + vocabulary_length) # use the smooth tech
                self.label_word_probability[lb][w] = math.log(prob) # take the logarithm

        self.label_class_prob = {lb: math.log(num/len(self.labels)) for lb, num in Counter(self.labels).items()} # take the logarithm

    def classify(self, test_data):
        word_set = set(t for t in test_data)
        max_prob = -math.inf
        result_label = None
        for lb in self.label_word_probability:
            sum_prob = self.label_class_prob[lb] # log(P(vj))
            for w in word_set: # log(P(x1,x2, ... | vj)) = log(P(x1 | vj)) + log(P(x2 | vj)) + ...
                if w in self.label_word_probability[lb]:
                    sum_prob += self.label_word_probability[lb][w]
                else:
                    sum_prob += self.label_missing_word_prob[lb]
            if sum_prob > max_prob:
                max_prob = sum_prob
                result_label = lb
        return result_label
