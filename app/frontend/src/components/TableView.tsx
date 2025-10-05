import {
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  useColorModeValue,
  Button,
  Box,
  Tooltip,
  HStack,
} from '@chakra-ui/react'
import { useState, useMemo } from 'preact/hooks'
import { TriToggle, TriToggleState } from './TriToggle'
import { GraphData } from '../types/GraphInterface'
import { TriangleUpIcon, TriangleDownIcon } from '@chakra-ui/icons'

interface TableViewProps {
  graphData: GraphData
  edgeLabel?: string
}

type SortOrder = 'asc' | 'desc' | null

export function TableView({ graphData, edgeLabel = 'label' }: TableViewProps) {
  const bgColor = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')
  const headerBg = useColorModeValue('gray.50', 'gray.700')
  const [showLabels, setShowLabels] = useState(false)
  const [sortOrder, setSortOrder] = useState<SortOrder>(null)

  // Track TriToggle state for each edge by edge ID
  const [edgeSelections, setEdgeSelections] = useState<Map<string, TriToggleState>>(new Map())

  // Create a map of node id to node label for quick lookup
  const nodeIdToLabel = new Map<string, string>()
  graphData.elements.nodes.forEach(node => {
    nodeIdToLabel.set(node.data.id, node.data.label)
  })

  const getNodeDisplay = (nodeId: string) => {
    if (showLabels) {
      return nodeIdToLabel.get(nodeId) || nodeId
    }
    return nodeId
  }

  // Sort edges by score
  const sortedEdges = useMemo(() => {
    const edges = [...graphData.elements.edges]

    if (!sortOrder) return edges

    return edges.sort((a, b) => {
      const scoreA = a.data.score
      const scoreB = b.data.score

      // Empty scores always at top
      if (scoreA === undefined || scoreA === null) return -1
      if (scoreB === undefined || scoreB === null) return 1

      const numA = Number(scoreA)
      const numB = Number(scoreB)

      if (sortOrder === 'asc') {
        return numA - numB
      } else {
        return numB - numA
      }
    })
  }, [graphData.elements.edges, sortOrder])

  const toggleSort = () => {
    setSortOrder(current => {
      if (current === null) return 'asc'
      if (current === 'asc') return 'desc'
      return null
    })
  }

  const emulateCuration = () => {
    const newSelections = new Map<string, TriToggleState>()

    // Get all edges with scores
    const edgesWithScores = graphData.elements.edges
      .map(edge => ({
        id: edge.data.id,
        score: edge.data.score !== undefined ? Number(edge.data.score) : null
      }))
      .filter(e => e.score !== null)

    if (edgesWithScores.length === 0) {
      // If no scores, select bottom 25% by index
      const numToSelect = Math.ceil(graphData.elements.edges.length * 0.25)
      graphData.elements.edges.slice(0, numToSelect).forEach(edge => {
        newSelections.set(edge.data.id, 'yes')
      })
    } else {
      // Sort by score and select bottom 25% (lowest scores)
      const sortedByScore = [...edgesWithScores].sort((a, b) => a.score! - b.score!)
      const threshold = Math.ceil(sortedByScore.length * 0.25)

      sortedByScore.slice(0, threshold).forEach(edge => {
        newSelections.set(edge.id, 'yes')
      })
    }

    setEdgeSelections(newSelections)
  }

  const handleToggleChange = (edgeId: string, newState: TriToggleState) => {
    setEdgeSelections(prev => {
      const updated = new Map(prev)
      updated.set(edgeId, newState)
      return updated
    })
  }

  return (
    <Box>
      <HStack mb={4} spacing={3}>
        <Button
          size="sm"
          colorScheme="teal"
          onClick={() => setShowLabels(!showLabels)}
        >
          Toggle Node Id/Name
        </Button>
        <Button
          size="sm"
          colorScheme="purple"
          onClick={toggleSort}
          rightIcon={
            sortOrder === 'asc' ? <TriangleUpIcon /> :
            sortOrder === 'desc' ? <TriangleDownIcon /> :
            undefined
          }
        >
          Sort by Score {sortOrder ? `(${sortOrder === 'asc' ? 'Ascending' : 'Descending'})` : ''}
        </Button>
        <Button
          size="sm"
          colorScheme="orange"
          onClick={emulateCuration}
        >
          Emulate Curation
        </Button>
      </HStack>
      <TableContainer
        borderRadius="lg"
        borderWidth="1px"
        borderColor={borderColor}
        bg={bgColor}
        shadow="xl"
        maxH="600px"
        overflowY="auto"
      >
        <Table variant="simple" size="md">
          <Thead bg={headerBg} position="sticky" top={0} zIndex={1}>
            <Tr>
              <Th maxW="400px">Source</Th>
              <Th maxW="400px">Label</Th>
              <Th maxW="400px">Target</Th>
              <Th maxW="400px">Score</Th>
              <Th maxW="400px">Selection</Th>
            </Tr>
          </Thead>
          <Tbody>
            {sortedEdges.map((edge) => {
              const sourceDisplay = getNodeDisplay(edge.data.source)
              const targetDisplay = getNodeDisplay(edge.data.target)
              const labelDisplay = edge.data[edgeLabel] || ''
              const scoreDisplay = edge.data.score !== undefined ? String(edge.data.score) : ''

              return (
                <Tr key={edge.data.id}>
                  <Td maxW="400px" overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap">
                    <Tooltip label={sourceDisplay} placement="top">
                      <Box>{sourceDisplay}</Box>
                    </Tooltip>
                  </Td>
                  <Td maxW="400px" overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap">
                    <Tooltip label={labelDisplay} placement="top">
                      <Box>{labelDisplay}</Box>
                    </Tooltip>
                  </Td>
                  <Td maxW="400px" overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap">
                    <Tooltip label={targetDisplay} placement="top">
                      <Box>{targetDisplay}</Box>
                    </Tooltip>
                  </Td>
                  <Td maxW="400px" overflow="hidden" textOverflow="ellipsis" whiteSpace="nowrap">
                    <Tooltip label={scoreDisplay} placement="top">
                      <Box>{scoreDisplay}</Box>
                    </Tooltip>
                  </Td>
                  <Td maxW="400px">
                    <TriToggle
                      state={edgeSelections.get(edge.data.id) || 'neutral'}
                      onChange={(newState) => handleToggleChange(edge.data.id, newState)}
                    />
                  </Td>
                </Tr>
              )
            })}
          </Tbody>
        </Table>
      </TableContainer>
    </Box>
  )
}
