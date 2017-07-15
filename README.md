# subspace_clustering
Code from a project to find newly-dense regions of categorical feature space using the technique here: https://www.cs.cornell.edu/johannes/papers/1998/sigmod1998-clique.pdf

The way this algorithm works is by choosing a fixed order of feature dimensionality (in this implementation, 2) and checking whether a given cell within the two-feature crosstab (for example: state = CA, channel = SEARCH) is "dense", where "dense" is a binary condition (of which a few were tested) defined as a function of historical data. This builds up into multivariable clusters by means of a graph: a clusters is defined as a fully connected clique in the graph defined by placing a link between mutually-dense category/values (For example: State=CA and channel=SEARCH would each be nodes on the graph, and if they had a density higher than our historical expectations, they would have a link).

At a high level, this method essentially looks for lower-dimensionality projections of higher-dimensional clusters, and identifies those clusters by seeing their "shadows" along every two-dimensional projection.


Code organization:
- cliques_graph: Contains the main object responsible for implementing the logic of the the mutual density graph
- expectation_objects: Defines the "Fitter" objects, which define a set of different criteria by which a cell may be
considered "dense"
- DimensionalPipeline: Creates a helper class to make the functions of time series aggregation (namely: window functions,
broadcasting, filling in values along time axes, etc) smoother to perform
