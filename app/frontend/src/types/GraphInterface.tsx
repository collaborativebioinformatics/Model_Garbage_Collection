export interface GraphData {
  elements: {
    nodes: Array<{
      data: {
        id: string
        label: string
        [key: string]: any
      }
      position?: {
        x: number
        y: number
      }
    }>
    edges: Array<{
      data: {
        id: string
        source: string
        target: string
        [key: string]: any
      }
    }>
  }
  data?: {
    title?: string
    description?: string
    tags?: string[]
  }
}
