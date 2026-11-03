import Box from '@mui/material/Box'
import Link from '@mui/material/Link'
import Typography from '@mui/material/Typography'

import {CodeBlock} from '../components/CodeBlock'
import {Prose} from '../components/Prose'
import {Section, type Band} from '../components/Section'
import {extend} from '../content/site'
import {lift, surfaces} from '../theme'

export function Extend({wash}: Band) {
  return (
    <Section id="extend" title={extend.title} lede={extend.lede} wash={wash}>
      <Box sx={{display: 'grid', gridTemplateColumns: {xs: '1fr', md: 'repeat(3, 1fr)'}, gap: {xs: 3, md: 3}}}>
        {extend.items.map((item) => (
          <Box key={item.title} sx={{display: 'flex', flexDirection: 'column', gap: 1.5, minWidth: 0}}>
            <Typography component="h3" variant="h4">
              {item.title}
            </Typography>
            <Typography variant="body2" sx={{color: 'text.secondary'}}>
              <Prose text={item.body} />
            </Typography>
            <Box
              sx={{
                ...lift,
                flex: 1,
                bgcolor: surfaces.code,
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 2,
                px: 2,
                py: 1.5,
              }}
            >
              <CodeBlock source={item.code} fontSize="0.8125rem" wrap />
            </Box>
            <Link href={item.link.href} variant="body2">
              {item.link.label}
            </Link>
          </Box>
        ))}
      </Box>
    </Section>
  )
}
