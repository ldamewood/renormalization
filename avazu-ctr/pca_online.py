from pyIPCA import CCIPCA, Hall_IPCA, Skocaj_IPCA
from scipy.sparse import csr_matrix
from sgd import data
from os.path import dirname, join

train = join(dirname(__file__),'train')
test = join(dirname(__file__),'test')

D = 2 ** 25
d = 2 ** 10
pcas = [
    CCIPCA(n_components=d),
    Hall_IPCA(n_components=d),
    Skocaj_IPCA(n_components=d)
]

for t, date, ID, x, y in data(train, D):
    X = csr_matrix((1,D))
    for i in x:
        X[0,i] = 1
    for pca in pcas:
        pca.fit(X)