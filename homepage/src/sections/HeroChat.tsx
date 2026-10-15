import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import Box from '@mui/material/Box'
import ButtonBase from '@mui/material/ButtonBase'
import Typography from '@mui/material/Typography'
import {useState} from 'react'

import {CodeBlock} from '../components/CodeBlock'
import {chart} from '../content/assets'
import {heroChat} from '../content/site'
import {glow, monoFamily, tokens} from '../theme'

const KNOCKOUT = tokens.mist

/**
 * A replay of Lumen answering a question, built from MUI and an inline SVG rather than an
 * iframe or a live LLM, so the most visible element on the page never waits on a model or
 * a demo deployment. Picking a question swaps in its recorded plan, SQL and chart.
 */
export function HeroChat() {
  const [index, setIndex] = useState(0)
  const q = heroChat.questions[index]

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: {xs: 'minmax(0, 1fr)', md: 'minmax(260px, 22rem) 1fr'},
        borderRadius: 3.5,
        overflow: 'hidden',
        bgcolor: tokens.lumen,
        // A wide soft glow so the panel reads as the lit part of the page.
        boxShadow: glow.ambient,
        // Both panes are saturated blue, so focus rings use the knockout grey instead.
        '--focus-ring': KNOCKOUT,
      }}
    >
      <Box sx={{bgcolor: tokens.lumenDeep, color: KNOCKOUT, px: {xs: 2.5, md: 3}, py: {xs: 3, md: 3.5}}}>
        <Typography variant="caption" component="p" sx={{opacity: 0.8, mb: 1.5}}>
          Ask about {heroChat.dataset}
        </Typography>
        <Box role="radiogroup" aria-label="Question" sx={{display: 'flex', flexDirection: 'column', gap: 1}}>
          {heroChat.questions.map((item, i) => (
            <ButtonBase
              key={item.prompt}
              role="radio"
              aria-checked={i === index}
              onClick={() => setIndex(i)}
              sx={{
                justifyContent: 'flex-start',
                textAlign: 'left',
                px: 1.75,
                py: 1.25,
                borderRadius: 2,
                fontSize: '0.9375rem',
                border: '1px solid',
                borderColor: i === index ? KNOCKOUT : 'rgba(238, 238, 238, 0.3)',
                bgcolor: i === index ? KNOCKOUT : 'transparent',
                color: i === index ? tokens.lumenDeep : KNOCKOUT,
                fontWeight: i === index ? 600 : 400,
                transition: 'background-color 160ms ease, color 160ms ease, box-shadow 200ms ease, border-color 200ms ease',
                '@media (hover: hover)': {'&:hover': {borderColor: KNOCKOUT, boxShadow: '0 0 18px rgba(238, 238, 238, 0.35)'}},
              }}
            >
              {item.prompt}
            </ButtonBase>
          ))}
        </Box>
        <Box component="ol" sx={{listStyle: 'none', p: 0, m: 0, mt: 3, display: 'flex', flexDirection: 'column', gap: 1.25}}>
          {heroChat.steps.map((step) => (
            <Box component="li" key={step.agent} sx={{display: 'flex', gap: 1, alignItems: 'flex-start'}}>
              <CheckCircleIcon sx={{fontSize: 18, mt: '2px'}} />
              <Typography variant="body2">
                <Box component="span" sx={{fontFamily: monoFamily, fontSize: '0.8125rem', opacity: 0.85}}>
                  {step.agent}
                </Box>{' '}
                {step.text}
              </Typography>
            </Box>
          ))}
        </Box>
      </Box>

      <Box sx={{color: KNOCKOUT, px: {xs: 2.5, md: 4}, py: {xs: 3, md: 3.5}, display: 'grid', gap: 2.5, alignContent: 'start', minWidth: 0}}>
        <Chart svg={chart(q.chart)} alt={q.alt} />
        <Box
          aria-label="Generated SQL"
          sx={{
            bgcolor: 'rgba(0, 0, 0, 0.18)',
            borderRadius: 2,
            px: 2,
            py: 1.5,
            // The scheme's code colours are tuned for the page, not the blue field.
            '--code-keyword': '#bbdefb',
            '--code-string': KNOCKOUT,
            '--code-call': KNOCKOUT,
            '& pre': {color: KNOCKOUT},
          }}
        >
          <CodeBlock source={q.sql} fontSize="0.75rem" wrap />
        </Box>
      </Box>
    </Box>
  )
}

/** The Vega-Lite chart Lumen drew, as inline SVG on the white plate the app itself uses. */
function Chart({svg, alt}: {svg: string; alt: string}) {
  return (
    <Box
      role="img"
      aria-label={alt}
      // Rendered from a spec this page ships, not from user input, so the markup is trusted.
      dangerouslySetInnerHTML={{__html: svg}}
      sx={{
        bgcolor: tokens.white,
        borderRadius: 2,
        px: 1.5,
        py: 1,
        '& svg': {display: 'block', width: '100%', height: 'auto'},
      }}
    />
  )
}
