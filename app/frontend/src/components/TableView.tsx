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
} from '@chakra-ui/react'
import { useState } from 'preact/hooks'
import { TriToggle } from './TriToggle'
import { GraphData } from '../types/GraphInterface'

interface TableViewProps {
  graphData: GraphData
  edgeLabel?: string
}

export function TableView({ graphData, edgeLabel = 'label' }: TableViewProps) {
  const bgColor = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')
  const headerBg = useColorModeValue('gray.50', 'gray.700')
  const [showLabels, setShowLabels] = useState(false)

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

  return (
    <Box>
      <Box mb={4}>
        <Button
          size="sm"
          colorScheme="teal"
          onClick={() => setShowLabels(!showLabels)}
        >
          Toggle Node Id/Name
        </Button>
      </Box>
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
            {graphData.elements.edges.map((edge) => {
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
                    <TriToggle defaultState="neutral" />
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
