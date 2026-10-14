import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import Box from '@mui/material/Box'
import ButtonBase from '@mui/material/ButtonBase'
import Typography from '@mui/material/Typography'
import {useState} from 'react'

import {CodeBlock} from '../components/CodeBlock'
import {heroChat} from '../content/site'
import {monoFamily, tokens} from '../theme'

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
                transition: 'background-color 160ms ease, color 160ms ease',
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

      <Box sx={{color: KNOCKOUT, px: {xs: 2.5, md: 4}, py: {xs: 3, md: 3.5}, display: 'grid', gap: 3, minWidth: 0}}>
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
          <CodeBlock source={q.sql} fontSize="0.8125rem" wrap />
        </Box>
        <BarChart title={q.title} bars={q.bars} />
      </Box>
    </Box>
  )
}

function BarChart({title, bars}: {title: string; bars: {label: string; value: number}[]}) {
  const max = Math.max(...bars.map((b) => b.value))
  return (
    <Box component="figure" sx={{m: 0}}>
      <Typography component="figcaption" sx={{fontWeight: 600, mb: 1.5}}>
        {title}
      </Typography>
      <Box sx={{display: 'grid', gridTemplateColumns: 'auto 1fr auto', columnGap: 1.5, rowGap: 1.25, alignItems: 'center'}}>
        {bars.map((b) => (
          <Box key={b.label} sx={{display: 'contents'}}>
            <Typography variant="body2">{b.label}</Typography>
            <Box sx={{height: 22, borderRadius: 1, bgcolor: 'rgba(0, 0, 0, 0.12)', overflow: 'hidden'}}>
              <Box
                sx={{
                  height: '100%',
                  width: `${(b.value / max) * 100}%`,
                  bgcolor: KNOCKOUT,
                  borderRadius: 1,
                  transition: 'width 320ms ease',
                }}
              />
            </Box>
            <Typography variant="body2" sx={{fontFamily: monoFamily, textAlign: 'right'}}>
              {b.value.toLocaleString('en-US')}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  )
}
