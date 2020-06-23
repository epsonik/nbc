class Point:

    def __init__(self, idx, vector, dist, preceding=None, following=None):
        self.idx = idx
        self.preceding = preceding
        self.following = following
        self.vector = vector
        self.dist = dist

    def __repr__(self):
        return "Point(idx: {}, vector: {}, dist: {}".format(self.idx, self.vector, self.dist)

    def __str__(self):
        return "idx: {}, vector: {}, dist: {}".format(self.idx, self.vector, self.dist)
