import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Container from '@mui/material/Container'
import Typography from '@mui/material/Typography'

import {CodeBlock} from '../components/CodeBlock'
import {InstallCommand} from '../components/InstallCommand'
import {hero, heroCode} from '../content/site'
import {surfaces} from '../theme'
import {HeroChat} from './HeroChat'

export function Hero() {
  return (
    <Box component="section" sx={{pt: {xs: 6, md: 9}, pb: {xs: 7, md: 10}}}>
      <Container maxWidth="lg">
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: {xs: 'minmax(0, 1fr)', md: '1.1fr 0.9fr'},
            columnGap: {md: 8},
            rowGap: 3.5,
            alignItems: 'start',
          }}
        >
          <Box>
            <Typography variant="h1" component="h1" sx={{maxWidth: '24ch'}}>
              {hero.title}
            </Typography>
            <Typography
              data-hero-tagline=""
              sx={{
                mt: 2.5,
                maxWidth: '30ch',
                fontSize: 'clamp(1.25rem, 1rem + 1.1vw, 1.75rem)',
                lineHeight: 1.35,
                fontWeight: 500,
                letterSpacing: '-0.015em',
                color: 'text.secondary',
              }}
            >
              {hero.tagline}
            </Typography>
          </Box>
          {/* Nudged down so the lede's first line sits on the headline's optical edge. */}
          <Box sx={{pt: {md: 1.25}}}>
            <Typography variant="body1" sx={{color: 'text.secondary', maxWidth: '38rem'}}>
              {hero.lede}
            </Typography>
            <Box sx={{display: 'flex', flexWrap: 'wrap', gap: 1.5, mt: 3.5, alignItems: 'center'}}>
              <Button variant="contained" size="large" href={hero.primary.href} sx={{flex: {xs: '1 1 100%', sm: '0 0 auto'}}}>
                {hero.primary.label}
              </Button>
              <Button
                size="large"
                variant="outlined"
                href={hero.secondary.href}
                sx={{color: 'text.primary', borderColor: 'divider', flex: {xs: '1 1 100%', sm: '0 0 auto'}}}
              >
                {hero.secondary.label}
              </Button>
            </Box>
            <Box sx={{mt: 2.5, maxWidth: '30rem'}}>
              <InstallCommand commands={{pip: hero.install, conda: hero.installConda}} />
            </Box>
          </Box>
        </Box>

        <Box sx={{mt: {xs: 6, md: 8}}}>
          <HeroChat />
        </Box>

        <Box
          sx={{
            mt: 3,
            bgcolor: surfaces.code,
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 2,
            overflow: 'hidden',
          }}
        >
          {heroCode.map((row, i) => (
            <Box
              key={row.label}
              sx={{
                display: 'grid',
                gridTemplateColumns: {xs: '1fr', md: '15rem minmax(0, 1fr)'},
                gap: {xs: 0.75, md: 3},
                px: {xs: 2, md: 3},
                py: {xs: 2, md: 2.25},
                borderTop: i === 0 ? 0 : '1px solid',
                borderColor: 'divider',
              }}
            >
              <Typography variant="body2" sx={{color: 'text.secondary'}}>
                {row.label}
              </Typography>
              <CodeBlock source={row.source} fontSize={{xs: '0.8125rem', md: '0.875rem'}} wrap />
            </Box>
          ))}
        </Box>
      </Container>
    </Box>
  )
}
