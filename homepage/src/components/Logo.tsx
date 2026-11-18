import Box from '@mui/material/Box'

import logo from '../assets/logo.svg'

/** Lumen's kite mark. The SVG carries its own colours, so it works on both schemes as is. */
export function Logo({size = 26, title}: {size?: number; title?: string}) {
  return (
    <Box
      component="img"
      src={logo}
      width={size}
      height={size}
      alt={title ?? ''}
      aria-hidden={title ? undefined : true}
      sx={{display: 'block'}}
    />
  )
}
