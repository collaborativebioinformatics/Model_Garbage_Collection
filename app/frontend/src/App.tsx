import {
  Box,
  Container,
  Heading,
  VStack,
  Text,
  SimpleGrid,
  Divider,
  useColorMode,
  Button,
  HStack,
} from '@chakra-ui/react'
import { useEffect, useState } from 'preact/hooks'
import { CounterCard } from './components/CounterCard'
import { UserCard } from './components/UserCard'
import { StatsCard } from './components/StatsCard'
import { GraphView } from './components/GraphView'
import { TriToggle } from './components/TriToggle'
import { TableView } from './components/TableView'
import { GraphData } from './types/GraphInterface'
import graphDataJson from './data/alzheimers_llm/graph.json'
import backboneDataJson from './data/alzheimers_llm/backbone_graph.json'

const graphData: GraphData = graphDataJson as GraphData
const backboneData: GraphData = backboneDataJson as GraphData

export function App() {
  const { setColorMode } = useColorMode()
  const [highlightedEdges, setHighlightedEdges] = useState<string[]>([])

  // Ensure dark mode is active
  useEffect(() => {
    setColorMode('dark')
  }, [setColorMode])

  const toggleEdgeHighlight = (edgeId: string) => {
    setHighlightedEdges(prev =>
      prev.includes(edgeId)
        ? prev.filter(id => id !== edgeId)
        : [...prev, edgeId]
    )
  }

  return (
    <Box minH="100vh" bg="gray.900" py={8}>
      <Container maxW="container.xl">
        <VStack spacing={8} align="stretch">
          {/* Header */}
          <Box textAlign="center" py={6}>
            <Heading
              size="2xl"
              bgGradient="linear(to-r, teal.300, purple.400, pink.400)"
              bgClip="text"
              mb={2}
            >
              Model Garbage Collection
            </Heading>
            <Text color="gray.400" fontSize="lg">
              Preact + Signals + Chakra UI + TypeScript
            </Text>
          </Box>

          <Divider borderColor="gray.700" />

          {/* Stats Section */}
          {/* <StatsCard /> */}

          {/* TriToggle Demo */}
          {/* <Box>
            <Heading size="md" color="teal.300" mb={4}>
              TriToggle Component
            </Heading>
            <TriToggle defaultState="neutral" onChange={(state) => console.log('TriToggle state:', state)} />
          </Box> */}

          

          {/* Graph Visualization */}
          <Box>
            <Heading size="md" color="teal.300" mb={4}>
              Network Graph Visualization
            </Heading>
            <GraphView
              graphData={graphData}
              highlightedEdges={highlightedEdges}
              edgeLabel="label"
            />
          </Box>


          {/* Backbone Table View */}
          <Box>
            <Heading size="md" color="teal.300" mb={4}>
              Edge Table View
            </Heading>
            <TableView graphData={backboneData} edgeLabel="label" />
          </Box>

          {/* Backbone Visualization */}
          <Box>
            <Heading size="md" color="teal.300" mb={4}>
              Backbone Query Graph Visualization
            </Heading>
            <GraphView
              graphData={backboneData}
              highlightedEdges={[]}
              edgeLabel="label"
            />
          </Box>

          {/* Interactive Components Grid */}
          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
            <CounterCard />
            <UserCard />
          </SimpleGrid>

          {/* Footer */}
          <Box textAlign="center" pt={8} pb={4}>
            <Text color="gray.500" fontSize="sm">
              Built with Vite, Preact, Chakra UI, Cytoscape.js, and Signals for state management
            </Text>
          </Box>
        </VStack>
      </Container>
    </Box>
  )
}
