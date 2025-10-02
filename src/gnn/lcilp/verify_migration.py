#!/usr/bin/env python3
"""
Migration Verification Script
Tests that all migrated modules can be imported without errors.
Run this after upgrading to PyTorch 2.1.2 and DGL 1.1.3.
"""

import sys
import traceback


def test_imports():
    """Test that all core modules can be imported."""

    results = []

    # Test core utilities
    print("Testing utils.graph_utils...")
    try:
        from utils.graph_utils import (
            ssp_multigraph_to_dgl,
            ssp_to_torch,
            send_graph_to_device,
            eccentricity,
        )

        results.append(("utils.graph_utils", "✓ PASS", None))
    except Exception as e:
        results.append(("utils.graph_utils", "✗ FAIL", str(e)))
        traceback.print_exc()

    # Test graph classifier
    print("Testing model.dgl.graph_classifier...")
    try:
        from model.dgl.graph_classifier import GraphClassifier

        results.append(("model.dgl.graph_classifier", "✓ PASS", None))
    except Exception as e:
        results.append(("model.dgl.graph_classifier", "✗ FAIL", str(e)))
        traceback.print_exc()

    # Test layers
    print("Testing model.dgl.layers...")
    try:
        from model.dgl.layers import RGCNLayer, RGCNBasisLayer

        results.append(("model.dgl.layers", "✓ PASS", None))
    except Exception as e:
        results.append(("model.dgl.layers", "✗ FAIL", str(e)))
        traceback.print_exc()

    # Test datasets
    print("Testing subgraph_extraction.datasets...")
    try:
        from subgraph_extraction.datasets import SubgraphDataset

        results.append(("subgraph_extraction.datasets", "✓ PASS", None))
    except Exception as e:
        results.append(("subgraph_extraction.datasets", "✗ FAIL", str(e)))
        traceback.print_exc()

    # Test RGCN model
    print("Testing model.dgl.rgcn_model...")
    try:
        from model.dgl.rgcn_model import RGCN

        results.append(("model.dgl.rgcn_model", "✓ PASS", None))
    except Exception as e:
        results.append(("model.dgl.rgcn_model", "✗ FAIL", str(e)))
        traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("MIGRATION VERIFICATION SUMMARY")
    print("=" * 60)

    for module, status, error in results:
        print(f"{module:40} {status}")
        if error:
            print(f"  Error: {error}")

    # Overall result
    failures = [r for r in results if "FAIL" in r[1]]
    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} module(s) failed to import")
        return 1
    else:
        print("RESULT: All modules imported successfully!")
        return 0


def test_versions():
    """Print version information."""
    print("=" * 60)
    print("VERSION INFORMATION")
    print("=" * 60)

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    try:
        import dgl

        print(f"DGL: {dgl.__version__}")
    except ImportError:
        print("DGL: NOT INSTALLED")

    try:
        import networkx as nx

        print(f"NetworkX: {nx.__version__}")
    except ImportError:
        print("NetworkX: NOT INSTALLED")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_versions()
    exit_code = test_imports()
    sys.exit(exit_code)
