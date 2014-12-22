#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from os import linesep
from collections import deque
import scipy.sparse
import numpy

#class Edge(object):
#    """
#    Weighted edge class. Java version by Sedgewick & Wayne:
#    http://algs4.cs.princeton.edu/43mst/Edge.java.html
#    """
#    v = 0
#    w = 0
#    weight = 0
#    
#    def __init__(self, v, w, weight):
#        if v < 0 or w < 0: raise IndexError("Vertex must be nonnegative")
#        self.v = v
#        self.w = w
#        self.weight = weight
#    
#    def other(self, v):
#        if   self.v == v: return self.w
#        elif self.w == v: return self.v
#        else:             raise IndexError()
#    
#    def __str__(self):
#        return "%d <--> %d : %f" % (self.v, self.w, self.weight)

class EdgeWeightedGraph(object):
    """
    Edge weighted graph. Java version by Sedgewick & Wayne:
    http://algs4.cs.princeton.edu/43mst/EdgeWeightedGraph.java.html
    Modified to do __setitem__ and __getitem__ faster.
    """

    _size = 0
    _edges = None
    _num_edges = 0
    _adj = None
    _csr_repr = None
    
    def __init__(self, V):
        if V < 0: raise IndexError("Number of vertices must be nonnegative")
        self._size = V
        self._edges = {}
        self._num_edges = 0
        self._adj = [[] for _ in range(V)]
        
    def __setitem__(self, key, value):
        """
        Set and get weight items using array notation.
        """
        # Do not accept slices or single entries
        if len(key) == 2 and type(key[0]) == int and type(key[1]) == int:
            i = key[0]
            j = key[1]
            added = False
            if not j in self._adj[i]:
                self._adj[i].append(j)
                self._edges[(i,j)] = value
                added = True
            if not i in self._adj[j]:
                self._adj[j].append(i)
                self._edges[(j,i)] = value            
                added = True
            if added:
                self._num_edges += 1
                # Remove cache of sparse matrix.
                self._csr_repr = None

    def __getitem__(self, key):
        if self._edges.has_key(key):
            return self._edges[key]
        return None
    
    def __delitem__(self, key):
        pass
    
    def adj(self, v):
        """
        Adjacent nodes.
        """
        return self._adj[v]
    
    def degree(self, v):
        """
        Number of neighbors.
        """
        return len(self._adj[v])
    
    def __str__(self):
        s = "vertices %d, edges %d" % (self._size, self._num_edges)
        for key, weight in self._edges.iteritems():
            if key[0] < key[1]:
                s += linesep + '%d <--> %d: %f' % (key[0], key[1], weight)
        return s
    
    @property
    def size(self):
        return self._size
    
    @property
    def edges(self):
        """
        Edge generator (no repeats).
        """
        for key,weight in self._edges:
            if key[0] < key[1] : yield (key, weight)

    @property
    def matrix(self):
        """
        Convert and cache scipy.sparse matrix representation.
        """
        if self._csr_repr == None:
            # TODO: make this faster by directly computing the csr matrix?
            lil_matrix = scipy.sparse.lil_matrix((self.size, self.size))
            for key, weight in self._edges.iteritems():
                lil_matrix[key] = weight
            self._csr_repr = lil_matrix.tocsr()
        return self._csr_repr

class Bipartite(object):
    """
    Bipartite class. Java version by Sedgewick & Wayne:
    http://algs4.cs.princeton.edu/41undirected/Bipartite.java.html
    
    Determines if a graph is bipartite and generates partition.
    """
    isBipartite = True
    _color = []
    
    def __init__(self, graph):
        self.isBipartite = True
        self._color = graph.size*[-1]

        # BFS to see if the graph is partitioned
        q = deque()
        q.append(0)
        self._color[0] = True
        
        while len(q) > 0:
            v = q.popleft()
            for w in graph.adj(v):
                if self._color[w] == -1:
                    self._color[w] = not self._color[v]
                    q.append(w)
                elif self._color[v] == self._color[w]:
                    self.isBipartite = False
    
    @property
    def color(self):
        """
        Binary array representing the partitions.
        """
        return self._color

    @property
    def partition(self):
        """
        Generator over partition elements.
        """
        if not self.isBipartite:
            raise ValueError("Graph is not bipartite")

        l = len(self._color[:])
        i = 0
        j = 0
        while i < l:
            if self._color[i]: yield i
            i += 1
        while j < l:
            if not self._color[j]: yield j
            j += 1
    
    def mask(self, choice):
        """
        numpy mask for selecting the partition.
        """
        if not self.isBipartite:
            raise ValueError("Graph is not bipartite")
        return numpy.nonzero(numpy.choose(self.color,[choice,not choice]))

    def __str__(self):
        if not self.isBipartite: return "Graph is not bipartite"
        ret = "Partitions:" + linesep
        ret += 'A: ' + ' '.join(['%d' % i for i in self.partition if self._color[i]])
        ret += linesep
        ret += 'B: ' + ' '.join(['%d' % i for i in self.partition if not self._color[i]])
        return ret

if __name__ == '__main__':
    """
    Unit tests.
    """
    g = EdgeWeightedGraph(6)
    g[0,1] = 1.
    g[1,2] = 1.
    g[2,3] = 1.
    g[3,4] = 1.
    g[4,5] = 1.
    g[5,0] = 1.

    print(g)
    print(g.matrix.todense())

    b = Bipartite(g)
    assert b.isBipartite == True    
    print(b)