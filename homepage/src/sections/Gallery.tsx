import Box from '@mui/material/Box'
import Link from '@mui/material/Link'
import Typography from '@mui/material/Typography'

import {Section, type Band} from '../components/Section'
import {Tile} from '../components/Tile'
import {gallery} from '../content/site'

export function Gallery({wash}: Band) {
  return (
    <Section id="gallery" title={gallery.title} lede={gallery.lede} wash={wash}>
      {gallery.groups.map((group, i) => (
        <Box key={group.title} sx={{mt: i ? {xs: 5, md: 6} : 0}}>
          <Typography component="h3" sx={{fontSize: '0.875rem', fontWeight: 600, mb: 2}}>
            {group.title}
          </Typography>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: {xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)'},
              gap: {xs: 3, md: 3},
            }}
          >
            {group.items.map((item) => (
              <Tile key={item.title} href={item.href} image={item.image} alt={`${item.title} screenshot`} title={item.title} caption={item.note} />
            ))}
          </Box>
        </Box>
      ))}
      <Link href={gallery.link.href} sx={{display: 'inline-block', mt: 4}}>
        {gallery.link.label}
      </Link>
    </Section>
  )
}
