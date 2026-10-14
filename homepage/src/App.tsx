import CssBaseline from '@mui/material/CssBaseline'
import InitColorSchemeScript from '@mui/material/InitColorSchemeScript'
import {ThemeProvider} from '@mui/material/styles'

import {Adoption} from './sections/Adoption'
import {BeyondTables} from './sections/BeyondTables'
import {Capabilities} from './sections/Capabilities'
import {Extend} from './sections/Extend'
import {Footer} from './sections/Footer'
import {Gallery} from './sections/Gallery'
import {Header} from './sections/Header'
import {Hero} from './sections/Hero'
import {Paths} from './sections/Paths'
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
        <Extend />
        <Paths wash />
        <Capabilities />
        <Adoption wash />
      </main>
      <Footer />
    </ThemeProvider>
  )
}
