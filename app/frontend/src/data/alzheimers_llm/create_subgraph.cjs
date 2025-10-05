#!/usr/bin/env node

const fs = require('fs');

// Read the subgraph.tsv file
const tsvContent = fs.readFileSync('subgraph.tsv', 'utf8');
const lines = tsvContent.trim().split('\n');

// Parse triples
const triples = lines.map(line => {
  const [source, label, target] = line.split('\t');
  return { source, label, target };
});

console.log(`Parsed ${triples.length} triples from subgraph.tsv`);

// Extract unique node IDs
const nodeIds = new Set();
triples.forEach(triple => {
  nodeIds.add(triple.source);
  nodeIds.add(triple.target);
});

console.log(`Found ${nodeIds.size} unique node IDs`);

// Read graph.json to get node information
const graphData = JSON.parse(fs.readFileSync('graph.json', 'utf8'));

// Create a map of node ID to node data
const nodeMap = new Map();
graphData.elements.nodes.forEach(node => {
  nodeMap.set(node.data.id, node.data);
});

console.log(`Loaded ${nodeMap.size} nodes from graph.json`);

// Build nodes array for subgraph
const nodes = [];
const missingNodes = [];

nodeIds.forEach(nodeId => {
  const nodeData = nodeMap.get(nodeId);
  if (nodeData) {
    nodes.push({
      data: { ...nodeData }
    });
  } else {
    // Create a placeholder node if not found in graph.json
    missingNodes.push(nodeId);
    nodes.push({
      data: {
        id: nodeId,
        label: nodeId,
        description: 'Node not found in original graph'
      }
    });
  }
});

if (missingNodes.length > 0) {
  console.log(`Warning: ${missingNodes.length} nodes not found in graph.json, created placeholders`);
}

// Build edges array
const edges = triples.map((triple, idx) => ({
  data: {
    id: `edge_${idx}`,
    source: triple.source,
    target: triple.target,
    label: triple.label
  }
}));

// Create the subgraph in GraphData format
const subgraph = {
  elements: {
    nodes: nodes,
    edges: edges
  },
  data: {
    title: "Alzheimer's LLM Subgraph",
    description: "Subgraph created from train.txt, test.txt, and valid.txt",
    tags: ["alzheimers", "subgraph", "llm"]
  }
};

// Write to subgraph.json
fs.writeFileSync('subgraph.json', JSON.stringify(subgraph, null, 2));

console.log(`✓ Created subgraph.json with ${nodes.length} nodes and ${edges.length} edges`);
