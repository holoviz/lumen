import CssBaseline from '@mui/material/CssBaseline'
import InitColorSchemeScript from '@mui/material/InitColorSchemeScript'
import {ThemeProvider} from '@mui/material/styles'

import {theme} from './theme'

export function App() {
  return (
    <ThemeProvider theme={theme} defaultMode="system">
      <InitColorSchemeScript attribute="data-color-scheme" defaultMode="system" />
      <CssBaseline />
      <main />
    </ThemeProvider>
  )
}
