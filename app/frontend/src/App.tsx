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

const sampleGraphData = {
  elements: {
    nodes: [
      { "data": { "id": "MONDO:0005723", "label": "faked", "description": "text definition" } },
      { "data": { "id": "MONDO:0006764", "label": "made_up", "description": "description2" } },
      { "data": { "id": "CHEBI:35610", "label": "label3", "description": "" } },
      { "data": { "id": "MONDO:0018908", "label": "label4", "description": "description3" } }
    ],
    edges: [
      { "data": { "id": "edge1", "source": "MONDO:0005723", "target": "MONDO:0006764", "label": "biolink:subclass_of" } },
      { "data": { "id": "edge2", "source": "CHEBI:35610", "target": "MONDO:0018908", "label": "biolink:treats_or_applied_or_studied_to_treat" } }
    ]
  },
  data: {
    title: "IT Infrastructure Network",
    description: "A sample network showing the connection between IT components.",
    tags: ["network", "example"]
  }
}

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
          <StatsCard />

          {/* TriToggle Demo */}
          <Box>
            <Heading size="md" color="teal.300" mb={4}>
              TriToggle Component
            </Heading>
            <TriToggle defaultState="neutral" onChange={(state) => console.log('TriToggle state:', state)} />
          </Box>

          {/* Table View */}
          <Box>
            <Heading size="md" color="teal.300" mb={4}>
              Edge Table View
            </Heading>
            <TableView graphData={sampleGraphData} edgeLabel="label" />
          </Box>

          {/* Graph Visualization */}
          <Box>
            <Heading size="md" color="teal.300" mb={4}>
              Network Graph Visualization
            </Heading>
            <GraphView
              graphData={sampleGraphData}
              highlightedEdges={highlightedEdges}
              edgeLabel="label"
            />
            <HStack mt={4} spacing={3}>
              <Button
                size="sm"
                colorScheme="red"
                onClick={() => toggleEdgeHighlight('desktop-server')}
              >
                Toggle Desktop-Server Edge
              </Button>
              <Button
                size="sm"
                colorScheme="red"
                onClick={() => toggleEdgeHighlight('server-database')}
              >
                Toggle Server-Database Edge
              </Button>
              <Button
                size="sm"
                colorScheme="gray"
                onClick={() => setHighlightedEdges([])}
              >
                Clear Highlights
              </Button>
            </HStack>
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
