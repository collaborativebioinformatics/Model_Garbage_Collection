import { render } from 'preact'
import { ChakraProvider, ColorModeScript } from '@chakra-ui/react'
import { App } from './App'
import theme from './theme'

render(
  <>
    <ColorModeScript initialColorMode={theme.config.initialColorMode} />
    <ChakraProvider theme={theme}>
      <App />
    </ChakraProvider>
  </>,
  document.getElementById('app')!
)
