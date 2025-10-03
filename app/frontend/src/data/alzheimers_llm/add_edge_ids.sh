#!/bin/bash

# Check if input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_graph.json>"
  exit 1
fi

INPUT_FILE="$1"

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: File '$INPUT_FILE' not found"
  exit 1
fi

# Add edge IDs using Node.js
node -e "
const fs = require('fs');
const inputFile = process.argv[1];
const data = JSON.parse(fs.readFileSync(inputFile, 'utf8'));

if (!data.elements || !data.elements.edges) {
  console.error('Error: Invalid graph structure - missing elements.edges');
  process.exit(1);
}

data.elements.edges = data.elements.edges.map((edge, idx) => ({
  ...edge,
  data: {
    id: \`edge_\${idx}\`,
    ...edge.data
  }
}));

fs.writeFileSync(inputFile, JSON.stringify(data, null, 2));
console.log('✓ Added', data.elements.edges.length, 'edge IDs to', inputFile);
" "$INPUT_FILE"
