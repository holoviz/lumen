import Box from '@mui/material/Box'
import Link from '@mui/material/Link'
import Typography from '@mui/material/Typography'

import {CodeBlock} from '../components/CodeBlock'
import {Section, type Band} from '../components/Section'
import {providers} from '../content/site'
import {lift, surfaces} from '../theme'

export function Providers({wash}: Band) {
  return (
    <Section id="providers" title={providers.title} lede={providers.lede} wash={wash}>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: {xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)'},
          gap: {xs: 3, md: 4},
        }}
      >
        {providers.groups.map((group) => (
          <Box key={group.title}>
            <Typography component="h3" sx={{fontSize: '0.875rem', fontWeight: 600, mb: 1.5}}>
              {group.title}
            </Typography>
            <Box component="ul" sx={{listStyle: 'none', p: 0, m: 0, display: 'flex', flexDirection: 'column', gap: 1}}>
              {group.names.map((name) => (
                <Typography component="li" variant="body2" key={name} sx={{color: 'text.secondary'}}>
                  {name}
                </Typography>
              ))}
            </Box>
          </Box>
        ))}
      </Box>

      <Box
        sx={{
          mt: {xs: 5, md: 7},
          display: 'grid',
          gridTemplateColumns: {xs: '1fr', md: '1fr 1.3fr'},
          gap: {xs: 2.5, md: 5},
          alignItems: 'center',
          ...lift,
          backgroundColor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 2,
          p: {xs: 2.5, md: 3.5},
        }}
      >
        <Box>
          <Typography component="h3" variant="h4">
            {providers.cli.title}
          </Typography>
          <Typography variant="body2" sx={{mt: 1.25, color: 'text.secondary'}}>
            {providers.cli.body}
          </Typography>
        </Box>
        <Box sx={{bgcolor: surfaces.code, border: '1px solid', borderColor: 'divider', borderRadius: 1.5, px: 2, py: 1.5, minWidth: 0}}>
          <CodeBlock source={providers.cli.code} fontSize="0.8125rem" wrap />
        </Box>
      </Box>
      <Link href={providers.link.href} sx={{display: 'inline-block', mt: 4}}>
        {providers.link.label}
      </Link>
    </Section>
  )
}
