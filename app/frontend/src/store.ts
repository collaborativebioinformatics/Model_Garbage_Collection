import { signal, computed } from '@preact/signals'

// Counter example using signals
export const count = signal(0)
export const doubleCount = computed(() => count.value * 2)

// User data example using deep signal
export const userData = signal({
  name: 'Guest User',
  notifications: 3,
  settings: {
    darkMode: true,
    emailNotifications: true
  }
})

// Actions
export const incrementCount = () => {
  count.value++
}

export const decrementCount = () => {
  count.value--
}

export const updateUserName = (name: string) => {
  userData.value = { ...userData.value, name }
}

export const clearNotifications = () => {
  userData.value = { ...userData.value, notifications: 0 }
}
