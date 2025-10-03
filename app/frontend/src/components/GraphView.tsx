import { Box, useColorModeValue, Tooltip, VStack, HStack, Text } from '@chakra-ui/react'
import { useEffect, useRef, useState, useMemo } from 'preact/hooks'
import cytoscape, { Core, NodeSingular, EdgeCollection } from 'cytoscape'
import { GraphData } from '../types/GraphInterface'

interface GraphViewProps {
  graphData: GraphData
  highlightedEdges?: string[]
  edgeLabel: string
  showNodeLabels?: boolean
}

// Color palette for edge labels
const COLOR_PALETTE = [
  '#4299E1', // blue
  '#48BB78', // green
  '#ED8936', // orange
  '#9F7AEA', // purple
  '#F56565', // red
  '#38B2AC', // teal
  '#ECC94B', // yellow
  '#ED64A6', // pink
  '#667EEA', // indigo
  '#FC8181', // red-light
]

// Legend component
interface LegendProps {
  colorMap: Map<string, string>
  bgColor: string
  borderColor: string
}

function Legend({ colorMap, bgColor, borderColor }: LegendProps) {
  return (
    <Box
      position="absolute"
      top="10px"
      right="10px"
      bg={bgColor}
      borderRadius="md"
      borderWidth="1px"
      borderColor={borderColor}
      p={3}
      shadow="md"
      maxH="200px"
      overflowY="auto"
      zIndex={1000}
    >
      <Text fontSize="sm" fontWeight="bold" mb={2}>Edge Types</Text>
      <VStack align="stretch" spacing={1}>
        {Array.from(colorMap.entries()).map(([label, color]) => (
          <HStack key={label} spacing={2}>
            <Box w="20px" h="3px" bg={color} borderRadius="sm" />
            <Text fontSize="xs">{label}</Text>
          </HStack>
        ))}
      </VStack>
    </Box>
  )
}

export function GraphView({ graphData, highlightedEdges = [], edgeLabel="label", showNodeLabels = true }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<Core | null>(null)
  const [hoveredNode, setHoveredNode] = useState<any>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  const bgColor = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')

  // Create color mapping for unique edge labels
  const edgeLabelColorMap = useMemo(() => {
    const uniqueLabels = new Set<string>()
    graphData.elements.edges.forEach(edge => {
      const labelValue = edge.data[edgeLabel]
      if (labelValue !== undefined && labelValue !== null) {
        uniqueLabels.add(String(labelValue))
      }
    })

    const colorMap = new Map<string, string>()
    Array.from(uniqueLabels).forEach((label, index) => {
      colorMap.set(label, COLOR_PALETTE[index % COLOR_PALETTE.length])
    })

    return colorMap
  }, [graphData, edgeLabel])


  useEffect(() => {
    if (!containerRef.current) return

    // Build dynamic styles for each edge label value
    const edgeStyles = Array.from(edgeLabelColorMap.entries()).map(([label, color]) => ({
      selector: `edge[${edgeLabel} = "${label}"]`,
      style: {
        'line-color': color,
        'target-arrow-color': color,
      }
    }))

    // Initialize Cytoscape
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements: graphData.elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#4299E1',
            'label': showNodeLabels ? 'data(label)' : '',
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
            'label': '',
            'font-size': '10px',
            'text-rotation': 'autorotate',
            'text-margin-y': -10,
            'color': '#A0AEC0'
          }
        },
        ...edgeStyles,
        {
          selector: 'edge.selected',
          style: {
            'label': `data(${edgeLabel})`,
            'width': 3,
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
        animationDuration: 5000,
        nodeRepulsion: 400000,
        idealEdgeLength: 100,
        edgeElasticity: 100,
        nestingFactor: 5,
        gravity: 80,
        numIter: 1000,
        initialTemp: 100,
        coolingFactor: 0.999,
        minTemp: 1.0
      }
    })

    // Initialize all edges with selected: false
    cyRef.current.edges().forEach(edge => {
      edge.data('selected', false)
    })

    // Add click event listener for edges
    cyRef.current.on('tap', 'edge', (evt: any) => {
      const edge = evt.target
      const currentSelected = edge.data('selected')

      // Toggle selected state
      edge.data('selected', !currentSelected)

      // Toggle CSS class for styling
      if (!currentSelected) {
        edge.addClass('selected')
      } else {
        edge.removeClass('selected')
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
  }, [graphData, edgeLabel, edgeLabelColorMap, showNodeLabels])

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

      <Legend colorMap={edgeLabelColorMap} bgColor={bgColor} borderColor={borderColor} />

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
