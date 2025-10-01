'''
Basic Chat interface

1. User asks LLM to extract relevant edges from a Knowledge Graph
2. LLM produces a list of Subject/Predicate/Object triples as a table
3. Users can augment SPO table with a column 'label' with True, Flase, None
4. Users can ask LLM to pipe SPOs to GNN Model and Train
5. LLM responds with label changes e.g. "edge X was true is now false"
'''