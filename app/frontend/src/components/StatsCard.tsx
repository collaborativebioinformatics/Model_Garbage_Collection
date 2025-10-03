import { Box, Grid, GridItem, Stat, StatLabel, StatNumber, StatHelpText, Icon } from '@chakra-ui/react'

// Simple icon placeholders (you can replace with actual icons)
const TrendIcon = () => <span>📈</span>
const UsersIcon = () => <span>👥</span>
const ActivityIcon = () => <span>⚡</span>

export function StatsCard() {
  return (
    <Box
      p={6}
      bg="gray.800"
      borderRadius="lg"
      borderWidth="1px"
      borderColor="gray.700"
      shadow="xl"
    >
      <Grid templateColumns="repeat(3, 1fr)" gap={6}>
        <GridItem>
          <Stat>
            <StatLabel color="gray.400">Total Models</StatLabel>
            <StatNumber color="cyan.400" fontSize="2xl">
              <TrendIcon /> 142
            </StatNumber>
            <StatHelpText color="green.400">↑ 12% from last month</StatHelpText>
          </Stat>
        </GridItem>

        <GridItem>
          <Stat>
            <StatLabel color="gray.400">Active Users</StatLabel>
            <StatNumber color="pink.400" fontSize="2xl">
              <UsersIcon /> 24
            </StatNumber>
            <StatHelpText color="green.400">↑ 5 new this week</StatHelpText>
          </Stat>
        </GridItem>

        <GridItem>
          <Stat>
            <StatLabel color="gray.400">Processing</StatLabel>
            <StatNumber color="orange.400" fontSize="2xl">
              <ActivityIcon /> 7
            </StatNumber>
            <StatHelpText color="gray.400">Currently active</StatHelpText>
          </Stat>
        </GridItem>
      </Grid>
    </Box>
  )
}
