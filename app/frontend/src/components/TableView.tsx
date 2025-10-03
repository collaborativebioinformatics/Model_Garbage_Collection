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
      >
        <Table variant="simple" size="md">
          <Thead bg={headerBg}>
            <Tr>
              <Th>Source</Th>
              <Th>Label</Th>
              <Th>Target</Th>
              <Th>Score</Th>
              <Th>Selection</Th>
            </Tr>
          </Thead>
          <Tbody>
            {graphData.elements.edges.map((edge) => (
              <Tr key={edge.data.id}>
                <Td>{getNodeDisplay(edge.data.source)}</Td>
                <Td>{edge.data[edgeLabel] || ''}</Td>
                <Td>{getNodeDisplay(edge.data.target)}</Td>
                <Td>{edge.data.score !== undefined ? edge.data.score : ''}</Td>
                <Td>
                  <TriToggle defaultState="neutral" />
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      </TableContainer>
    </Box>
  )
}
