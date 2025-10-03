import {
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  TableContainer,
  useColorModeValue,
} from '@chakra-ui/react'
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

  return (
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
              <Td>{edge.data.source}</Td>
              <Td>{edge.data[edgeLabel] || ''}</Td>
              <Td>{edge.data.target}</Td>
              <Td>{edge.data.score !== undefined ? edge.data.score : ''}</Td>
              <Td>
                <TriToggle defaultState="neutral" />
              </Td>
            </Tr>
          ))}
        </Tbody>
      </Table>
    </TableContainer>
  )
}
