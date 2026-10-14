import CssBaseline from '@mui/material/CssBaseline'
import InitColorSchemeScript from '@mui/material/InitColorSchemeScript'
import {ThemeProvider} from '@mui/material/styles'

import {BeyondTables} from './sections/BeyondTables'
import {Footer} from './sections/Footer'
import {Header} from './sections/Header'
import {Hero} from './sections/Hero'
import {HowItWorks} from './sections/HowItWorks'
import {Providers} from './sections/Providers'
import {theme} from './theme'

export function App() {
  return (
    <ThemeProvider theme={theme} defaultMode="system">
      <InitColorSchemeScript attribute="data-color-scheme" defaultMode="system" />
      <CssBaseline />
      <Header />
      <main>
        <Hero />
        <HowItWorks wash />
        <BeyondTables />
        <Providers wash />
      </main>
      <Footer />
    </ThemeProvider>
  )
}
