# Model Garbage Collection - Frontend

A modern web application built with Preact, TypeScript, Vite, and Chakra UI.

## Tech Stack

- **Preact** - Fast 3kB alternative to React
- **TypeScript** - Type-safe JavaScript
- **Vite** - Next generation build tool
- **Chakra UI** - Component library with dark theme
- **Cytoscape.js** - Graph visualization library
- **@preact/signals** - Reactive state management
- **Yarn** - Package manager

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- Yarn package manager

### Installation & Running

1. **Navigate to the frontend directory**
   ```bash
   cd app/frontend
   ```

2. **Install dependencies**
   ```bash
   yarn install
   ```

3. **Start the development server**
   ```bash
   yarn dev
   ```

4. **Open your browser**

   The app will be available at `http://localhost:5173`

   You should see:
   - Statistics dashboard
   - Interactive graph visualization with force-directed layout
   - Counter and user profile components

### Build for Production

```bash
yarn build
```

### Preview Production Build

```bash
yarn preview
```

## Features

- 🌙 Dark theme by default
- 📊 Interactive graph visualization with Cytoscape.js
- 🎯 Force-directed graph layout
- 🔍 Node hover tooltips showing all properties
- 🔴 Edge highlighting functionality
- ⚡ Signals-based state management
- 🎨 Chakra UI component library
- 📱 Responsive design
- 🔥 Hot Module Replacement (HMR)
- 📦 Optimized production builds with Vite

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── CounterCard.tsx    # Counter with signals demo
│   │   ├── UserCard.tsx       # User profile component
│   │   ├── StatsCard.tsx      # Statistics display
│   │   └── GraphView.tsx      # Graph visualization component
│   ├── store.ts               # Signals state management
│   ├── theme.ts               # Chakra UI theme configuration
│   ├── App.tsx                # Main app component
│   └── main.tsx               # App entry point
├── index.html                 # HTML template
├── vite.config.ts             # Vite configuration
└── tsconfig.json              # TypeScript configuration
```

## Graph Visualization

The `GraphView` component accepts graph data in the following format:

```typescript
{
  elements: {
    nodes: [
      {
        data: { id: "node1", label: "Node 1", /* custom properties */ },
        position: { x: 100, y: 100 }  // optional
      }
    ],
    edges: [
      {
        data: {
          id: "edge1",
          source: "node1",
          target: "node2",
          interaction: "relationship"  // optional label
        }
      }
    ]
  }
}
```

**Features:**
- Automatic force-directed layout (cose algorithm)
- Hover over nodes to see all properties
- Pass `highlightedEdges` prop to highlight specific edges by ID
- Fully customizable styling
