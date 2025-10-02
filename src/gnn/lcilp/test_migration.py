#!/usr/bin/env python
"""
Migration Test Script for PyTorch 2.1.2 / DGL 1.1.3

Tests that all migrated APIs work correctly with the new versions.
"""

import sys
import torch
import dgl
import numpy as np
import scipy.sparse as ssp

print("=" * 60)
print("PyTorch/DGL Migration Test")
print("=" * 60)

# Test 1: Check versions
print("\n1. Version Check")
print(f"   PyTorch: {torch.__version__}")
print(f"   DGL: {dgl.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

assert torch.__version__.startswith("2.1"), (
    f"Expected PyTorch 2.1.x, got {torch.__version__}"
)
assert dgl.__version__.startswith("1.1"), f"Expected DGL 1.1.x, got {dgl.__version__}"
print("   ✓ Versions correct")

# Test 2: Test dgl.readout_nodes (replaces mean_nodes)
print("\n2. Testing dgl.readout_nodes (replaces mean_nodes)")
g = dgl.graph(([0, 1, 2], [1, 2, 3]))
g.ndata["repr"] = torch.randn(4, 10)
result = dgl.readout_nodes(g, "repr", op="mean")
expected = g.ndata["repr"].mean(dim=0)
assert torch.allclose(result, expected, atol=1e-6), (
    "readout_nodes mean operation failed"
)
print("   ✓ dgl.readout_nodes works")

# Test 3: Test dgl.from_networkx (replaces DGLGraph constructor)
print("\n3. Testing dgl.from_networkx (replaces DGLGraph)")
import networkx as nx

g_nx = nx.MultiDiGraph()
g_nx.add_nodes_from([0, 1, 2])
g_nx.add_edges_from([(0, 1, {"type": 0}), (1, 2, {"type": 1})])
g_dgl = dgl.from_networkx(g_nx, edge_attrs=["type"])
assert g_dgl.num_nodes() == 3, "Graph has wrong number of nodes"
assert g_dgl.num_edges() == 2, "Graph has wrong number of edges"
assert "type" in g_dgl.edata, "Edge type attribute missing"
print("   ✓ dgl.from_networkx works")

# Test 4: Test subgraph with edata[dgl.EID]
print("\n4. Testing subgraph with edata[dgl.EID] (replaces parent_eid)")
g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]))
g.edata["type"] = torch.tensor([0, 1, 0, 1])
sub_g = g.subgraph([0, 1, 2])
original_types = g.edata["type"][sub_g.edata[dgl.EID]]
assert len(original_types) == sub_g.num_edges(), "Edge type extraction failed"
print("   ✓ subgraph with edata[dgl.EID] works")

# Test 5: Test torch.sparse_coo_tensor (replaces torch.sparse.FloatTensor)
print("\n5. Testing torch.sparse_coo_tensor (replaces torch.sparse.FloatTensor)")
row = np.array([0, 1, 2])
col = np.array([1, 2, 0])
data = np.array([1.0, 2.0, 3.0])
idx = torch.tensor([row, col], dtype=torch.long)
dat = torch.tensor(data, dtype=torch.float)
A = torch.sparse_coo_tensor(idx, dat, size=(3, 3))
assert A.is_sparse, "Tensor should be sparse"
assert A.shape == (3, 3), "Tensor has wrong shape"
print("   ✓ torch.sparse_coo_tensor works")

# Test 6: Test nn.Identity (replaces custom Identity class)
print("\n6. Testing nn.Identity (replaces custom Identity class)")
import torch.nn as nn

identity = nn.Identity()
x = torch.randn(5, 10)
y = identity(x)
assert torch.equal(x, y), "nn.Identity should return input unchanged"
print("   ✓ nn.Identity works")

# Test 7: Test tensor construction (replaces LongTensor/FloatTensor)
print("\n7. Testing torch.tensor with dtype (replaces LongTensor/FloatTensor)")
x = torch.tensor([1, 2, 3], dtype=torch.long)
assert x.dtype == torch.long, "Long tensor creation failed"
y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
assert y.dtype == torch.float32, "Float tensor creation failed"
print("   ✓ torch.tensor with dtype works")

# Test 8: Test graph attribute iteration (replaces attr_schemes)
print("\n8. Testing graph attribute iteration (replaces attr_schemes)")
g = dgl.graph(([0, 1], [1, 2]))
g.ndata["feat"] = torch.randn(3, 5)
g.edata["weight"] = torch.randn(2)
ndata_keys = list(g.ndata.keys())
edata_keys = list(g.edata.keys())
assert "feat" in ndata_keys, "Node data iteration failed"
assert "weight" in edata_keys, "Edge data iteration failed"
print("   ✓ Graph attribute iteration works")

# Test 9: Test NetworkX .nodes() (replaces .nbunch_iter())
print("\n9. Testing NetworkX .nodes() (replaces .nbunch_iter())")
G = nx.Graph()
G.add_nodes_from([0, 1, 2])
nodes_list = list(G.nodes())
assert len(nodes_list) == 3, "Node iteration failed"
print("   ✓ NetworkX .nodes() works")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nMigration to PyTorch 2.1.2 / DGL 1.1.3 successful!")
