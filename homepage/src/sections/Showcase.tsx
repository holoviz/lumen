import Box from '@mui/material/Box'
import ButtonBase from '@mui/material/ButtonBase'
import Typography from '@mui/material/Typography'
import {useState} from 'react'

import {Section, type Band} from '../components/Section'
import {showcase} from '../content/site'
import {tokens} from '../theme'

/** One large screenshot at a time, picked by its question, so each stays readable. */
export function Showcase({wash}: Band) {
  const [index, setIndex] = useState(0)
  const shot = showcase.items[index]

  return (
    <Section id="showcase" title={showcase.title} lede={showcase.lede} wash={wash}>
      <Box role="tablist" aria-label="Example answers" sx={{display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2.5}}>
        {showcase.items.map((item, i) => (
          <ButtonBase
            key={item.name}
            role="tab"
            aria-selected={i === index}
            onClick={() => setIndex(i)}
            sx={{
              px: 1.75,
              py: 1,
              borderRadius: 2,
              fontSize: '0.9375rem',
              border: '1px solid',
              borderColor: i === index ? 'primary.main' : 'divider',
              bgcolor: i === index ? 'primary.main' : 'background.paper',
              color: i === index ? 'primary.contrastText' : 'text.primary',
              fontWeight: i === index ? 600 : 400,
            }}
          >
            {item.title}
          </ButtonBase>
        ))}
      </Box>
      <Box component="figure" role="tabpanel" sx={{m: 0}}>
        <Box
          component="img"
          src={shot.image}
          alt={`Lumen answering: ${shot.question}`}
          width={1600}
          height={1000}
          loading="lazy"
          decoding="async"
          sx={{
            display: 'block',
            width: '100%',
            height: 'auto',
            // Screenshots of the light app, so the plate stays white in dark mode too.
            backgroundColor: tokens.white,
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 2,
          }}
        />
        <Typography component="figcaption" variant="body2" sx={{mt: 1.5, color: 'text.secondary'}}>
          “{shot.question}”
        </Typography>
      </Box>
    </Section>
  )
}
