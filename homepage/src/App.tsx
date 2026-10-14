import CssBaseline from '@mui/material/CssBaseline'
import InitColorSchemeScript from '@mui/material/InitColorSchemeScript'
import {ThemeProvider} from '@mui/material/styles'

import {BeyondTables} from './sections/BeyondTables'
import {Footer} from './sections/Footer'
import {Gallery} from './sections/Gallery'
import {Header} from './sections/Header'
import {Hero} from './sections/Hero'
import {HowItWorks} from './sections/HowItWorks'
import {Providers} from './sections/Providers'
import {Sources} from './sections/Sources'
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
        <Gallery wash />
        <Providers />
        <Sources wash />
      </main>
      <Footer />
    </ThemeProvider>
  )
}
