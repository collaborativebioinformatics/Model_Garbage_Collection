import { Box, useColorModeValue, Tooltip } from '@chakra-ui/react'
import { useEffect, useRef, useState } from 'preact/hooks'
import cytoscape, { Core, NodeSingular, EdgeCollection } from 'cytoscape'

interface GraphData {
  elements: {
    nodes: Array<{
      data: {
        id: string
        label: string
        [key: string]: any
      }
      position?: {
        x: number
        y: number
      }
    }>
    edges: Array<{
      data: {
        id: string
        source: string
        target: string
        [key: string]: any
      }
    }>
  }
  data?: {
    title?: string
    description?: string
    tags?: string[]
  }
}

interface GraphViewProps {
  graphData: GraphData
  highlightedEdges?: string[]
}

export function GraphView({ graphData, highlightedEdges = [] }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<Core | null>(null)
  const [hoveredNode, setHoveredNode] = useState<any>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  const bgColor = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')

  useEffect(() => {
    if (!containerRef.current) return

    // Initialize Cytoscape
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements: graphData.elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#4299E1',
            'label': 'data(label)',
            'color': '#fff',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '12px',
            'width': '40px',
            'height': '40px',
            'text-outline-width': 2,
            'text-outline-color': '#2D3748',
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#718096',
            'target-arrow-color': '#718096',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(interaction)',
            'font-size': '10px',
            'text-rotation': 'autorotate',
            'text-margin-y': -10,
            'color': '#A0AEC0'
          }
        },
        {
          selector: 'edge.highlighted',
          style: {
            'width': 4,
            'line-color': '#F56565',
            'target-arrow-color': '#F56565',
            'z-index': 999
          }
        },
        {
          selector: 'node:hover',
          style: {
            'background-color': '#38B2AC',
            'width': '50px',
            'height': '50px',
          }
        }
      ],
      layout: {
        name: 'cose', // Force-directed layout
        animate: true,
        animationDuration: 1000,
        nodeRepulsion: 400000,
        idealEdgeLength: 100,
        edgeElasticity: 100,
        nestingFactor: 5,
        gravity: 80,
        numIter: 1000,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0
      }
    })

    // Add hover event listeners
    cyRef.current.on('mouseover', 'node', (evt: any) => {
      const node: NodeSingular = evt.target
      const nodeData = node.data()
      const renderedPos = node.renderedPosition()

      setHoveredNode(nodeData)
      setTooltipPos({ x: renderedPos.x, y: renderedPos.y })
    })

    cyRef.current.on('mouseout', 'node', () => {
      setHoveredNode(null)
    })

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy()
      }
    }
  }, [graphData])

  // Update highlighted edges when prop changes
  useEffect(() => {
    if (!cyRef.current) return

    // Remove all highlighting
    cyRef.current.edges().removeClass('highlighted')

    // Add highlighting to specified edges
    if (highlightedEdges.length > 0) {
      highlightedEdges.forEach(edgeId => {
        const edge = cyRef.current!.getElementById(edgeId) as EdgeCollection
        if (edge.length > 0) {
          edge.addClass('highlighted')
        }
      })
    }
  }, [highlightedEdges])

  return (
    <Box position="relative" width="100%" height="600px">
      <Box
        ref={containerRef}
        width="100%"
        height="100%"
        bg={bgColor}
        borderRadius="lg"
        borderWidth="1px"
        borderColor={borderColor}
        shadow="xl"
      />

      {hoveredNode && (
        <Box
          position="absolute"
          top={`${tooltipPos.y - 80}px`}
          left={`${tooltipPos.x + 20}px`}
          bg="gray.900"
          color="white"
          p={3}
          borderRadius="md"
          shadow="xl"
          maxW="300px"
          zIndex={1000}
          pointerEvents="none"
        >
          <Box fontWeight="bold" mb={2}>{hoveredNode.label}</Box>
          {Object.entries(hoveredNode).map(([key, value]) => (
            key !== 'id' && key !== 'label' && (
              <Box key={key} fontSize="sm">
                <Box as="span" color="cyan.300">{key}:</Box>{' '}
                <Box as="span" color="gray.300">
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </Box>
              </Box>
            )
          ))}
        </Box>
      )}
    </Box>
  )
}
