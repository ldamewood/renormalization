#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from os import linesep
from scipy.sparse import lil_matrix
from numpy import nonzero, choose
from collections import deque

class EdgeWeightedGraph(object):
    """
    Undirected graph with edge weights. Original Java version by Sedgewick & 
    Wayne [http://algs4.cs.princeton.edu/43mst/EdgeWeightedGraph.java.html].
    Modified to do __setitem__ and __getitem__ faster by using a dict for the
    edges.
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
        """Setter for weight values."""
        # Do not accept slices or single entries
        if len(key) == 2 and type(key[0]) == int and type(key[1]) == int:
            i = key[0]
            j = key[1]
            added = False            
            if j not in self._adj[i]:
                self._adj[i].append(j)
                added = True
            if i not in self._adj[j]:
                self._adj[j].append(i)          
                added = True
            if added:
                self._num_edges += 1
            
            self._edges[(j,i)] = value  
            self._edges[(i,j)] = value
            # Remove cache of sparse matrix.
            self._csr_repr = None

    def __getitem__(self, key):
        """Weight value getter."""
        if self._edges.has_key(key):
            return self._edges[key]
        return None
    
    def __delitem__(self, key):
        pass
    
    def adj(self, v):
        """Adjacent nodes."""
        return self._adj[v]
    
    def degree(self, v):
        """Number of neighbors."""
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
        """Edge generator (no repeats)."""
        for key,weight in self._edges:
            if key[0] < key[1] : yield (key, weight)

    @property
    def matrix(self):
        """Convert and cache scipy.sparse matrix representation."""
        if self._csr_repr == None:
            # TODO: make this faster by directly computing the csr matrix?
            lil = lil_matrix((self.size, self.size))
            for key, weight in self._edges.iteritems():
                lil[key] = weight
            self._csr_repr = lil.tocsr()
        return self._csr_repr

class Bipartite(object):
    """
    Bipartite class. Original Java version by Sedgewick & Wayne
    [http://algs4.cs.princeton.edu/41undirected/Bipartite.java.html].
    
    Determines if a graph is bipartite and generates partition.
    """
    isBipartite = True
    _color = []
    
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
        for w in graph.adj(v):
            if colors[w] == -1:
                colors[w] = (colors[v] + 1) % n
                q.append(w)
            elif colors[v] == colors[w]:
                raise ValueError("Graph cannot be %d-colored" % n)
                
    return colors

if __name__ == '__main__':
    """Unit tests."""
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