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
import {HowItWorks} from './sections/HowItWorks'
import {Paths} from './sections/Paths'
import {Providers} from './sections/Providers'
import {Showcase} from './sections/Showcase'
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
        <Showcase wash />
        <HowItWorks />
        <BeyondTables wash />
        <Gallery />
        <Providers wash />
        <Sources />
        <Extend wash />
        <Paths />
        <Capabilities wash />
        <Adoption />
      </main>
      <Footer />
    </ThemeProvider>
  )
}
