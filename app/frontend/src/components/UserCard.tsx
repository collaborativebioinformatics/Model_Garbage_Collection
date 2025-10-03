import {
  Box,
  Badge,
  Button,
  Heading,
  Input,
  Text,
  VStack,
  HStack,
} from '@chakra-ui/react'
import { useState } from 'preact/hooks'
import { userData, updateUserName, clearNotifications } from '../store'

export function UserCard() {
  const [inputName, setInputName] = useState('')

  const handleUpdateName = () => {
    if (inputName.trim()) {
      updateUserName(inputName)
      setInputName('')
    }
  }

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
        <HStack justify="space-between">
          <Heading size="md" color="purple.300">
            User Profile
          </Heading>
          {userData.value.notifications > 0 && (
            <Badge colorScheme="red" fontSize="sm">
              {userData.value.notifications} new
            </Badge>
          )}
        </HStack>

        <Text fontSize="lg">
          Welcome, <Text as="span" fontWeight="bold" color="purple.400">
            {userData.value.name}
          </Text>
        </Text>

        <VStack spacing={2} align="stretch">
          <Input
            placeholder="Enter new name"
            value={inputName}
            onInput={(e) => setInputName((e.target as HTMLInputElement).value)}
            bg="gray.700"
            borderColor="gray.600"
            _hover={{ borderColor: 'purple.400' }}
            _focus={{ borderColor: 'purple.400', boxShadow: '0 0 0 1px var(--chakra-colors-purple-400)' }}
          />
          <HStack spacing={2}>
            <Button
              colorScheme="purple"
              size="sm"
              onClick={handleUpdateName}
              flex={1}
            >
              Update Name
            </Button>
            <Button
              colorScheme="red"
              size="sm"
              onClick={clearNotifications}
              flex={1}
              isDisabled={userData.value.notifications === 0}
            >
              Clear Notifications
            </Button>
          </HStack>
        </VStack>
      </VStack>
    </Box>
  )
}
