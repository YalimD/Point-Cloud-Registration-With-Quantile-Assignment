import math
import queue

import numpy as np
from scipy.sparse import csr_matrix


class HopcroftKarp(object):

    def __init__(self, cost_matrix):

        self.u_length = cost_matrix.shape[0]
        self.max_length = max(cost_matrix.shape)
        self.NIL = self.max_length

        self.adjacency = csr_matrix(cost_matrix)

        self.pairs = np.zeros((2, self.max_length + 1), dtype=np.int)
        self.pairs[:][:] = self.NIL
        self.dist = np.zeros(self.NIL + 1)

        self.queue = queue.SimpleQueue()

    def fill(self):
        for u in range(self.u_length):
            if self.pairs[0][u] == self.NIL:
                self.dist[u] = 0
                self.queue.put(u)
            else:
                self.dist[u] = math.inf

    def bfs(self) -> bool:

        self.fill()

        self.dist[self.NIL] = math.inf
        while not self.queue.empty():
            u = self.queue.get()
            if self.dist[u] < self.dist[self.NIL]:
                for v in self.adjacency[u].indices:
                    u_match = self.pairs[1][v]
                    if self.dist[u_match] == math.inf:
                        self.dist[u_match] = self.dist[u] + 1
                        self.queue.put(u_match)

        return self.dist[self.NIL] != math.inf

    def dfs(self, u) -> bool:

        if u != self.NIL:
            for v in self.adjacency[u].indices:
                if self.dist[self.pairs[1][v]] == self.dist[u] + 1:
                    if self.dfs(self.pairs[1][v]):
                        self.pairs[1][v] = u
                        self.pairs[0][u] = v
                        return True
            self.dist[u] = math.inf
            return False
        return True

    def match(self):

        while self.bfs():
            for u in range(self.u_length):
                if self.pairs[0][u] == self.NIL:
                    self.dfs(u)

        match_indices = np.argwhere(self.pairs[:][0] < self.NIL).T[0]

        return match_indices, self.pairs[0][match_indices]


if __name__ == "__main__":
    test_matrix = np.asarray([
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
    ])

    matches = HopcroftKarp(test_matrix).match()
    print(matches[0])
    print(matches[1])
