import math
from enum import Enum
from collections import Counter
from collections import defaultdict

class TF_Scale(Enum):
    RAW = 1
    SUB_LINEAR = 2
    MAXIMUM = 3

class VectorSpaceModel(object):
    rawDF = defaultdict(int)
    _IDF = None
    alpha = 0.2 # for maximun tf scaling

    def __init__(self, words):
        self.rawTF = Counter(words)
        try:
            self._maxCountInTF = max(self.rawTF.values())
        except ValueError:
            self._maxCountInTF = 1 # 0 can't be the divisor

    def calWeight(self, tf_method):
        if not self._IDF:
            calIDF()
        self.vector = {}
        for k in self.rawTF:
            self.vector[k] = self.calTF(k, self, tf_method)*self._IDF[k]

    def normalize(self):
        if not self.vector:
            self.calWeight(TF_Scale.SUB_LINEAR)
        norm = math.sqrt(sum(weight**2 for weight in self.vector.values()))
        for k in self.vector:
            self.vector[k] /= norm
        self._isUnit = True
        return self

    def dot(self, other):
        return VectorSpaceModel.dotProduct(self, other)

    def getTerms(self):
        return self.rawTF.keys()

    @staticmethod
    def dotProduct(a, b):
        if not(a._isUnit and b._isUnit):
            raise "Please use toUnit() to normalize all vectors."
        result = 0
        for k in a.vector.keys():
            if k in b.vector:
                result += a.vector[k]*b.vector[k]
        return result

    @staticmethod
    def calTF(t, vsm, tf_method):
        if tf_method == TF_Scale.SUB_LINEAR:
            return 1 + math.log(vsm.rawTF[t]) if vsm.rawTF[t] > 0 else 0
        elif tf_method == TF_Scale.MAXIMUM:
            return vsm.alpha + (1 - vsm.alpha)*vsm.rawTF[t]/vsm._maxCountInTF
        else: #TF_Scale.RAW
            return vsm.rawTF[t]

    @classmethod
    def getCorpus(cls):
        return cls.rawDF.keys()

    @classmethod
    def accumulateDocumentFrequency(cls, words):
        for w in words:
            cls.rawDF[w] += 1

    @classmethod
    def calIDF(cls):
        N = len(cls.rawDF)
        cls._IDF = {k:math.log(N/v) for k, v in cls.rawDF.items()}
