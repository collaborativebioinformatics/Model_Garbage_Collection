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

const sampleGraphData = {
  elements: {
    nodes: [
      {
        data: {
          id: "desktop",
          label: "Desktop",
          type: "device"
        },
        position: { x: 100, y: 100 }
      },
      {
        data: {
          id: "server",
          label: "Server",
          type: "device"
        },
        position: { x: 300, y: 100 }
      },
      {
        data: {
          id: "database",
          label: "Database",
          type: "database"
        },
        position: { x: 300, y: 300 }
      }
    ],
    edges: [
      {
        data: {
          id: "desktop-server",
          source: "desktop",
          target: "server",
          interaction: "requests"
        }
      },
      {
        data: {
          id: "server-database",
          source: "server",
          target: "database",
          interaction: "query"
        }
      }
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

          {/* Graph Visualization */}
          <Box>
            <Heading size="md" color="teal.300" mb={4}>
              Network Graph Visualization
            </Heading>
            <GraphView
              graphData={sampleGraphData}
              highlightedEdges={highlightedEdges}
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
