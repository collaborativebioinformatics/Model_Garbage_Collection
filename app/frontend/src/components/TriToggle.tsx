import { Button } from '@chakra-ui/react'
import { useState } from 'preact/hooks'
import { CheckIcon, CloseIcon } from '@chakra-ui/icons'

export type TriToggleState = 'neutral' | 'yes' | 'no'

interface TriToggleProps {
  defaultState?: TriToggleState
  onChange?: (state: TriToggleState) => void
}

export function TriToggle({ defaultState = 'neutral', onChange }: TriToggleProps) {
  const [state, setState] = useState<TriToggleState>(defaultState)

  const handleClick = () => {
    const nextState =
      state === 'neutral' ? 'yes' :
      state === 'yes' ? 'no' :
      'neutral'

    setState(nextState)
    onChange?.(nextState)
  }

  const getIcon = () => {
    switch (state) {
      case 'yes':
        return <CheckIcon color="green.500" />
      case 'no':
        return <CloseIcon color="red.500" />
      case 'neutral':
      default:
        return (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <rect x="2" y="2" width="12" height="12" stroke="currentColor" strokeWidth="2" fill="none" rx="2" />
          </svg>
        )
    }
  }

  const getColorScheme = () => {
    switch (state) {
      case 'yes':
        return 'green'
      case 'no':
        return 'red'
      case 'neutral':
      default:
        return 'gray'
    }
  }

  return (
    <Button
      onClick={handleClick}
      colorScheme={getColorScheme()}
      variant="outline"
      size="md"
      leftIcon={getIcon()}
    >
      {state.charAt(0).toUpperCase() + state.slice(1)}
    </Button>
  )
}
