import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Link from '@mui/material/Link'
import Typography from '@mui/material/Typography'

import {CodeBlock} from '../components/CodeBlock'
import {Section, type Band} from '../components/Section'
import {beyondTables} from '../content/site'
import {lift, surfaces} from '../theme'

/** N-dimensional and geospatial data, side by side, each with the question and SQL it answers. */
export function BeyondTables({wash}: Band) {
  return (
    <Section id="beyond-tables" title={beyondTables.title} lede={beyondTables.lede} wash={wash}>
      <Box sx={{display: 'grid', gridTemplateColumns: {xs: '1fr', md: 'repeat(2, 1fr)'}, gap: {xs: 3, md: 4}}}>
        {beyondTables.items.map((item) => (
          <Box
            key={item.title}
            sx={{
              ...lift,
              backgroundColor: 'background.paper',
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 2,
              p: {xs: 2.5, md: 3.5},
              display: 'flex',
              flexDirection: 'column',
              gap: 2,
              minWidth: 0,
            }}
          >
            <Typography component="h3" variant="h3">
              {item.title}
            </Typography>
            <Box sx={{display: 'flex', flexWrap: 'wrap', gap: 0.75}}>
              {item.formats.map((f) => (
                <Chip key={f} label={f} size="small" variant="outlined" />
              ))}
            </Box>
            <Typography variant="body2" sx={{color: 'text.secondary'}}>
              {item.body}
            </Typography>
            <Box sx={{bgcolor: surfaces.code, border: '1px solid', borderColor: 'divider', borderRadius: 1.5, px: 2, py: 1.5}}>
              <CodeBlock source={item.code} fontSize="0.8125rem" wrap />
            </Box>
            <Box sx={{flex: 1}}>
              <Typography variant="body2" sx={{fontWeight: 600, mb: 1}}>
                “{item.question}”
              </Typography>
              <Box sx={{bgcolor: surfaces.code, border: '1px solid', borderColor: 'divider', borderRadius: 1.5, px: 2, py: 1.5}}>
                <CodeBlock source={item.sql} fontSize="0.8125rem" wrap />
              </Box>
            </Box>
            <Link href={item.link.href}>{item.link.label}</Link>
          </Box>
        ))}
      </Box>
    </Section>
  )
}
