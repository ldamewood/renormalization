#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from os import linesep
from scipy.sparse import csr_matrix
from numpy import nonzero, choose
from collections import deque

class Graph(object):
    """
    Fixed size graph object with integer vertices and generic vertex and edge
    values.
    """
    
    def __init__(self, V):
        if V < 0: raise IndexError('Size of graph must be nonnegative')
        self._size = V
        self._num_edges = 0
        self._matrix_cache = None
        self._adj = [[] for _ in range(V)]
        self._edges = {}
        self._vertices = {}
    
    def _check_vertex(self, key):
        """Check validity of vertex key"""
        if hasattr(key, '__iter__'): return False
        try:
            key = int(key)
        except TypeError:
            return False
    
        if key < 0 or key >= self._size:
            raise IndexError('key value %d is out of range.' % key)

        return True
    
    def _check_edge(self, key):
        """Check validity of edge key"""
        if not hasattr(key, '__iter__'): return False
        try:
            length = len(key)
        except TypeError:
            return False

        if length != 2: return False
        
        if not self._check_vertex(key[0]): return False
        if not self._check_vertex(key[1]): return False
        
        return True
    
    def __setitem__(self, key, value):
        """Setter for vertex and edge values."""
        if self._check_vertex(key):
            self._vertices[key] = value
        elif self._check_edge(key):
            self.addEdge(key[0], key[1])
            self._matrix_cache = None # Remove cache of sparse matrix.
            self._edges[key] = value
        else:
            raise TypeError('Not a valid key')

    def __getitem__(self, key):
        """Getter for vertex and edge values."""
        if self._check_vertex(key) and key in self._vertices:
            return self._vertices[key]
        elif self._check_edge(key) and key in self._edges:
            return self._edges[key]
        return None
    
    def __delitem__(self, key):
        # Never delete items.
        pass
    
    def addEdge(self, fr, to):
        # Add edge without adding value
        fr = int(fr)
        to = int(to)
        if to not in self._adj[fr]:
            self._adj[fr].append(to)
            self._num_edges += 1
    
    @property
    def adj(self):
        """Adjacency matrix nodes."""
        return self._adj
    
    def degree(self, v):
        """Number of neighbors."""
        return len(self._adj[v])
    
    @property
    def size(self):
        return self._size
    
    @property
    def edgeWeights(self):
        """Edge generator (no repeats)."""
        for key,value in self._edges:
            yield (key, value)

    @property
    def matrix(self):
        """Convert and cache scipy.sparse matrix representation."""
        if self._matrix_cache == None:
            rows = []
            cols = []
            data = []
            for key, value in self._edges.iteritems():
                rows.append(key[0])
                cols.append(key[1])
                data.append(value)
            self._matrix_cache = csr_matrix((data, (rows, cols)), shape = (self.size, self.size))
        return self._matrix_cache

    def __str__(self):
        s = "vertices %d, edges %d" % (self._size, self._num_edges)
        for key, value in self._vertices.iteritems():
            s += linesep + 'Node %d: ' % key
            s += str(value)
        for key, weight in self._edges.iteritems():
            s += linesep + '%d --> %d: ' % (key[0], key[1])
            s += str(weight)
        return s
    
class UndirectedGraph(Graph):
    def __init__(self, V):
        super(self.__class__, self).__init__(V)
    
    def __setitem__(self, key, value):
        """Setter for vertex and edge values."""
        if self._check_vertex(key):
            self._vertices[key] = value
        elif self._check_edge(key):
            self.addEdge(key[0], key[1])
            self._matrix_cache = None # Remove cache of sparse matrix.
            self._edges[(key[0], key[1])] = value
            self._edges[(key[1], key[0])] = value
        else:
            raise TypeError('Not a valid key')
    
    def addEdge(self, a, b):
        super(self.__class__, self).addEdge(a,b)
        super(self.__class__, self).addEdge(b,a)
    
    def __str__(self):
        s = "vertices %d, edges %d" % (self._size, self._num_edges)
        for key, weight in self._edges.iteritems():
            if key[0] < key[1]:
                s += linesep + '%d <--> %d: %f' % (key[0], key[1], weight)
        return s

class Bipartite(object):
    """
    Bipartite class. Original Java version by Sedgewick & Wayne
    [http://algs4.cs.princeton.edu/41undirected/Bipartite.java.html].
    
    Determines if a graph is bipartite and generates partition.
    """
    
    def __init__(self, graph):
        self._color = graph.size*[-1]
        try:
            self._color = ncolor(graph, 2)
        except ValueError:
            self.isBipartite = False
        else:
            self.isBipartite = True

    @property
    def color(self):
        """Binary array representing the partitions."""
        return self._color

    @property
    def partition(self):
        """Generator over partition elements."""
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
        """Create mask for selecting the partition."""
        if not self.isBipartite:
            raise ValueError("Graph is not bipartite")
        return nonzero(choose(self.color,[not choice,choice]))

    def __str__(self):
        if not self.isBipartite: return "Graph is not bipartite"
        ret = "Partitions:" + linesep
        ret += 'A: ' + ' '.join(['%d' % i for i in self.partition if self._color[i]])
        ret += linesep
        ret += 'B: ' + ' '.join(['%d' % i for i in self.partition if not self._color[i]])
        return ret

def ncolor(graph, n):
    """
    Greedy algorithm for coloring a graph with n colors.
    """
    colors = graph.size * [-1]
    q = deque()
    q.append(0)
    colors[0] = 0
    
    # BFS: label colors as we go.
    while len(q) > 0:
        v = q.popleft()
        for w in graph.adj[v]:
            if colors[w] == -1:
                colors[w] = (colors[v] + 1) % n
                q.append(w)
            elif colors[v] == colors[w]:
                raise ValueError("Graph cannot be %d-colored" % n)
                
    return colors

if __name__ == '__main__':
    """Unit tests."""
    g = Graph(6)
    # Add directed edge weights
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