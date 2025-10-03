import { Box, Button, Heading, HStack, Text, VStack } from '@chakra-ui/react'
import { count, doubleCount, incrementCount, decrementCount } from '../store'

export function CounterCard() {
  return (
    <Box
      p={6}
      bg="gray.800"
      borderRadius="lg"
      borderWidth="1px"
      borderColor="gray.700"
      shadow="xl"
    >
      <VStack spacing={4} align="stretch">
        <Heading size="md" color="teal.300">
          Counter with Signals
        </Heading>

        <Text fontSize="xl">
          Count: <Text as="span" fontWeight="bold" color="teal.400">{count.value}</Text>
        </Text>

        <Text fontSize="md" color="gray.400">
          Double Count: {doubleCount.value}
        </Text>

        <HStack spacing={3}>
          <Button
            colorScheme="teal"
            onClick={decrementCount}
            size="sm"
          >
            Decrement
          </Button>
          <Button
            colorScheme="teal"
            onClick={incrementCount}
            size="sm"
          >
            Increment
          </Button>
        </HStack>
      </VStack>
    </Box>
  )
}
