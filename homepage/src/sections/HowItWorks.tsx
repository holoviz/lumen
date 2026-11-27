import Box from '@mui/material/Box'
import Link from '@mui/material/Link'
import Typography from '@mui/material/Typography'

import {Section, type Band} from '../components/Section'
import {howItWorks} from '../content/site'
import {kites, lift, monoFamily} from '../theme'

const ACCENTS = [kites.blue, kites.purple, kites.red, kites.orange]

/** The four steps behind every answer, numbered because they always run in this order. */
export function HowItWorks({wash}: Band) {
  return (
    <Section id="how-it-works" title={howItWorks.title} lede={howItWorks.lede} wash={wash}>
      <Box
        component="ol"
        sx={{
          listStyle: 'none',
          p: 0,
          m: 0,
          display: 'grid',
          gridTemplateColumns: {xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)'},
          gap: {xs: 2.5, md: 3},
        }}
      >
        {howItWorks.steps.map((step, i) => (
          <Box
            component="li"
            key={step.title}
            sx={{
              ...lift,
              backgroundColor: 'background.paper',
              border: '1px solid',
              borderColor: 'divider',
              borderTop: `3px solid ${ACCENTS[i]}`,
              borderRadius: 2,
              p: {xs: 2.5, md: 3},
              display: 'flex',
              flexDirection: 'column',
              gap: 1.25,
            }}
          >
            <Typography variant="caption" sx={{color: 'text.secondary'}}>
              Step {i + 1}
            </Typography>
            <Typography component="h3" variant="h4">
              {step.title}
            </Typography>
            <Typography variant="body2" sx={{color: 'text.secondary', flex: 1}}>
              {step.body}
            </Typography>
            <Box component="code" sx={{fontFamily: monoFamily, fontSize: '0.8125rem', color: 'primary.main'}}>
              {step.code}
            </Box>
          </Box>
        ))}
      </Box>
      <Link href={howItWorks.link.href} sx={{display: 'inline-block', mt: 4}}>
        {howItWorks.link.label}
      </Link>
    </Section>
  )
}
