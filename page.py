from graph import Graph
import numpy

def pagerank(internet, err = 1.e-10, damping = 0.85, order = 1):
    """Original PageRank via power iterator with damping"""
    g = Graph(internet.size)
    for v in range(internet.size):
        d = len(internet.adj[v])
        for w in internet.adj[v]:
            g[(w,v)] = 1./d
    
    r = numpy.random.normal(size = internet.size)
    r /= numpy.linalg.norm(r, order)
    r = numpy.ones(internet.size) / internet.size
    e = numpy.ones(internet.size) * float('inf')
    while numpy.linalg.norm(e, order) > err ** (1./order):
        last_r = r.copy()
        # update with teleportation
        r = damping * g.matrix.dot(r) + (1-damping)/internet.size
        r /= numpy.linalg.norm(r, order)
        e = r - last_r
    return r

if __name__ == '__main__':
    names = ['y','a','m']
    g1 = Graph(3)
    g1.addEdge(0,0)
    g1.addEdge(0,1)
    g1.addEdge(1,0)
    g1.addEdge(1,2)
    g1.addEdge(2,2)
    
    pr = pagerank(g1)
    print(numpy.argsort(pr)[::-1])
    
    g2 = Graph(11)
    g2.addEdge(1,2)
    g2.addEdge(2,1)
    g2.addEdge(3,0)
    g2.addEdge(3,1)
    g2.addEdge(4,1)
    g2.addEdge(4,3)
    g2.addEdge(4,5)
    g2.addEdge(5,1)
    g2.addEdge(5,4)
    g2.addEdge(6,1)
    g2.addEdge(6,4)
    g2.addEdge(7,1)
    g2.addEdge(7,4)
    g2.addEdge(8,1)
    g2.addEdge(8,4)
    g2.addEdge(9,4)
    g2.addEdge(10,4)
    
    pr = pagerank(g2)
    print(numpy.argsort(pr)[::-1])