import Box from '@mui/material/Box'
import Link from '@mui/material/Link'
import Typography from '@mui/material/Typography'

import {CodeBlock} from '../components/CodeBlock'
import {Section, type Band} from '../components/Section'
import {sources} from '../content/site'
import {lift, surfaces} from '../theme'

export function Sources({wash}: Band) {
  return (
    <Section id="sources" title={sources.title} lede={sources.lede} wash={wash}>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: {xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)'},
          gap: {xs: 1.5, md: 2},
        }}
      >
        {sources.items.map((item) => (
          <Box
            key={item.name}
            sx={{
              ...lift,
              backgroundColor: 'background.paper',
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 2,
              px: 2.5,
              py: 2,
            }}
          >
            <Typography component="h3" sx={{fontSize: '0.9375rem', fontWeight: 600}}>
              {item.name}
            </Typography>
            <Typography variant="caption" component="p" sx={{color: 'text.secondary', mt: 0.25}}>
              {item.note}
            </Typography>
          </Box>
        ))}
      </Box>
      <Box sx={{mt: 3, bgcolor: surfaces.code, border: '1px solid', borderColor: 'divider', borderRadius: 2, px: {xs: 2, md: 3}, py: 2}}>
        <CodeBlock source={sources.code} fontSize={{xs: '0.8125rem', md: '0.875rem'}} wrap />
      </Box>
      <Link href={sources.link.href} sx={{display: 'inline-block', mt: 4}}>
        {sources.link.label}
      </Link>
    </Section>
  )
}
