#!/usr/bin/env node

const fs = require('fs');

// Read subgraph.json
const subgraph = JSON.parse(fs.readFileSync('subgraph.json', 'utf8'));

// Function to generate random score
function generateScore() {
  const isNegative = Math.random() < 0.1; // 10% negative

  if (isNegative) {
    // Negative: between -20 and -5
    return (Math.random() * 15 - 20).toFixed(2);
  } else {
    // Positive: between 10 and 20
    return (Math.random() * 10 + 10).toFixed(2);
  }
}

// Add scores to all edges
subgraph.elements.edges.forEach(edge => {
  edge.data.score = parseFloat(generateScore());
});

// Count statistics
const scores = subgraph.elements.edges.map(e => e.data.score);
const negativeCount = scores.filter(s => s < 0).length;
const positiveCount = scores.filter(s => s >= 0).length;

console.log(`Added scores to ${subgraph.elements.edges.length} edges`);
console.log(`Positive scores: ${positiveCount} (${(positiveCount/scores.length*100).toFixed(1)}%)`);
console.log(`Negative scores: ${negativeCount} (${(negativeCount/scores.length*100).toFixed(1)}%)`);
console.log(`Score range: ${Math.min(...scores).toFixed(2)} to ${Math.max(...scores).toFixed(2)}`);

// Write back to subgraph.json
fs.writeFileSync('subgraph.json', JSON.stringify(subgraph, null, 2));

console.log(`✓ Updated subgraph.json with scores`);
