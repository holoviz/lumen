import Box from '@mui/material/Box'
import ButtonBase from '@mui/material/ButtonBase'
import Typography from '@mui/material/Typography'
import {useEffect, useRef, useState} from 'react'

import {Section, type Band} from '../components/Section'
import {showcase} from '../content/site'
import {glow, tokens} from '../theme'

const FRAME = {
  display: 'block',
  width: '100%',
  height: 'auto',
  // Screenshots of the light app, so the plate stays white in dark mode too.
  backgroundColor: tokens.white,
  border: '1px solid',
  borderColor: 'divider',
  borderRadius: 2,
  boxShadow: glow.soft,
} as const

/**
 * A silent screen recording that plays while it is on screen. Anyone who asked for reduced
 * motion gets the poster and the player controls instead, and nothing starts by itself.
 */
function Clip({clip, poster, label}: {clip: {mp4: string; webm: string}; poster: string; label: string}) {
  const ref = useRef<HTMLVideoElement>(null)
  const [still, setStill] = useState(false)

  useEffect(() => {
    const video = ref.current
    if (!video) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    setStill(reduced)
    if (reduced) return
    const watcher = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) video.play().catch(() => undefined)
      else video.pause()
    }, {threshold: 0.4})
    watcher.observe(video)
    return () => watcher.disconnect()
  }, [clip])

  return (
    <Box
      component="video"
      ref={ref}
      poster={poster}
      muted
      loop
      playsInline
      preload="metadata"
      controls={still}
      aria-label={label}
      sx={FRAME}
    >
      <source src={clip.webm} type="video/webm" />
      <source src={clip.mp4} type="video/mp4" />
    </Box>
  )
}

/** One large recording or screenshot at a time, picked by its title, so each stays readable. */
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
              transition: 'box-shadow 200ms ease, border-color 200ms ease',
              '@media (hover: hover)': {'&:hover': {borderColor: 'primary.main', boxShadow: glow.hover}},
            }}
          >
            {item.title}
          </ButtonBase>
        ))}
      </Box>
      <Box component="figure" role="tabpanel" sx={{m: 0}}>
        {shot.clip ? (
          <Clip key={shot.name} clip={shot.clip} poster={shot.image} label={shot.caption} />
        ) : (
          <Box
            component="img"
            src={shot.image}
            alt={shot.caption}
            width={1600}
            height={1000}
            loading="lazy"
            decoding="async"
            sx={FRAME}
          />
        )}
        <Typography component="figcaption" variant="body2" sx={{mt: 1.5, color: 'text.secondary'}}>
          {shot.caption}
        </Typography>
      </Box>
    </Section>
  )
}
