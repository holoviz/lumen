import Box from '@mui/material/Box'
import Typography from '@mui/material/Typography'
import type {ReactNode} from 'react'

import {glow, tokens} from '../theme'

/** A linked 16:9 screenshot with a title and a caption, as used by the gallery. */
export function Tile({
  href,
  image,
  alt,
  title,
  caption,
}: {
  href: string
  image: string
  alt: string
  title: string
  caption: ReactNode
}) {
  return (
    <Box
      component="a"
      href={href}
      data-tile=""
      sx={{
        textDecoration: 'none',
        color: 'inherit',
        display: 'flex',
        flexDirection: 'column',
        '&:hover .tile-title': {color: 'primary.main'},
        '@media (hover: hover)': {
          '&:hover img': {borderColor: 'primary.main', boxShadow: glow.hover, transform: 'translateY(-3px)'},
        },
      }}
    >
      <Box
        component="img"
        src={image}
        alt={alt}
        // Matches the 960px wide renders, so the box is reserved before the image arrives.
        width={960}
        height={540}
        loading="lazy"
        decoding="async"
        sx={{
          display: 'block',
          width: '100%',
          // The attributes above would otherwise win over the aspect ratio.
          height: 'auto',
          aspectRatio: '16 / 9',
          objectPosition: 'top',
          objectFit: 'cover',
          // Screenshots of light apps, so the plate stays white in dark mode too.
          backgroundColor: tokens.white,
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 1,
          transition: 'transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease',
        }}
      />
      <Typography
        component="h3"
        className="tile-title"
        sx={{mt: 1.25, fontSize: '0.9375rem', fontWeight: 600, transition: 'color 160ms ease'}}
      >
        {title}
      </Typography>
      <Typography variant="caption" component="p" sx={{color: 'text.secondary', mt: 0.25}}>
        {caption}
      </Typography>
    </Box>
  )
}
