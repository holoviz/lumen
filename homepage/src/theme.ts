import {createTheme} from '@mui/material/styles'

/**
 * Design tokens for lumen.holoviz.org.
 *
 * The primary blue is the one the Zensical docs use (docs/assets/extra.css), so moving from
 * this page into the docs does not change the accent. The four kite colours come from the
 * logo mark (docs/assets/logo.svg) and are kept for small accents only.
 */
export const tokens = {
  /** Docs primary. */
  lumen: '#1976D2',
  /** Docs primary, darkened. Pressed states and inverted panes. */
  lumenDeep: '#1565C0',
  /** Light grey used on blue surfaces. */
  mist: '#eeeeee',
  /** The faint blue-grey behind alternating bands. */
  wash: '#f3f6f9',
  /** One step lighter than the wash, for code sitting on a white surface. */
  codeWash: '#f7f9fb',
  ink: '#000000',
  slate: '#5f6368',
  white: '#ffffff',
} as const

/** The logo's four kites, mid stops of each gradient. */
export const kites = {
  blue: '#0277BD',
  purple: '#AB47BC',
  red: '#EF5350',
  orange: '#F57C00',
} as const

/** Dark scheme, derived by darkening the primary blue rather than reaching for a neutral black. */
const dark = {
  bg: '#0b1524',
  /** Cards and panes. Sits above both the page and the wash. */
  surface: '#14243a',
  /** The dark counterpart of tokens.wash: a band, one step off the page, below the cards. */
  wash: '#0f1c2e',
  border: 'rgba(238, 238, 238, 0.16)',
  primary: '#64b5f6',
  text: '#eef2f5',
  textSecondary: '#93aec0',
} as const

/**
 * Surfaces that are not palette roles.
 *
 * `background.default` is the page and `background.paper` is a card, which leaves no role for
 * "a band one step off the page". These are custom properties rather than theme values so a
 * band follows the colour scheme without any component subscribing to it.
 */
export const surfaces = {
  wash: 'var(--surface-wash)',
  code: 'var(--surface-code)',
} as const

/**
 * The keyboard focus ring, as an inherited custom property with the accent as its fallback.
 * Declaring the fallback inside the var() rather than on the focusable element itself is what
 * lets an inverted surface reset --focus-ring for everything inside it: a custom property
 * declared on the element would beat the ancestor's value.
 */
const FOCUS_RING = 'var(--focus-ring, var(--mui-palette-primary-main))'

const fontFamily = 'Inter, system-ui, Arial, Helvetica, sans-serif'
const monoFamily = '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace'

/**
 * Glow and lift for cards. The colour is the scheme's own primary, read from the CSS variable MUI
 * exposes, so the same rule glows light blue on the dark page and deep blue on the white one.
 * The lift only runs where hover exists, so touch screens never get a stuck raised card.
 */
const PRIMARY = 'var(--mui-palette-primary-mainChannel)'

export const glow = {
  hover: `0 0 0 1px rgb(${PRIMARY} / 0.5), 0 14px 38px rgb(${PRIMARY} / 0.28)`,
  ambient: `0 24px 70px rgb(${PRIMARY} / 0.24)`,
  soft: `0 10px 40px rgb(${PRIMARY} / 0.14)`,
} as const

export const lift = {
  transition: 'transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease',
  '@media (hover: hover)': {
    '&:hover': {transform: 'translateY(-3px)', borderColor: 'primary.main', boxShadow: glow.hover},
  },
} as const

export const theme = createTheme({
  cssVariables: {colorSchemeSelector: 'data-color-scheme'},
  colorSchemes: {
    light: {
      palette: {
        primary: {main: tokens.lumen, dark: tokens.lumenDeep, contrastText: tokens.white},
        // The page is white and bands are washed, rather than a grey page with white cards:
        // grey never appears as a full-page field, so the blue hero is the brightest thing here.
        background: {default: tokens.white, paper: tokens.white},
        text: {primary: tokens.ink, secondary: tokens.slate},
        divider: 'rgba(0, 0, 0, 0.10)',
      },
    },
    dark: {
      palette: {
        primary: {main: dark.primary, dark: tokens.lumen, contrastText: '#06101c'},
        background: {default: dark.bg, paper: dark.surface},
        text: {primary: dark.text, secondary: dark.textSecondary},
        divider: dark.border,
      },
    },
  },
  // The logo's corners are generously rounded relative to its size. Large panes echo
  // that; controls stay tighter so they read as controls.
  shape: {borderRadius: 8},
  typography: {
    fontFamily,
    // Inter is loaded as a variable font with an optical size axis, so display sizes get
    // the tighter, higher-contrast cut and body text stays at the reading cut.
    h1: {
      fontFamily,
      fontSize: 'clamp(2.5rem, 1.35rem + 3.6vw, 4rem)',
      lineHeight: 1.02,
      fontWeight: 700,
      letterSpacing: '-0.032em',
      fontVariationSettings: '"opsz" 32',
    },
    h2: {
      fontFamily,
      fontSize: 'clamp(1.75rem, 1.2rem + 1.8vw, 2.5rem)',
      lineHeight: 1.1,
      fontWeight: 650,
      letterSpacing: '-0.022em',
      fontVariationSettings: '"opsz" 32',
    },
    h3: {fontSize: '1.5rem', lineHeight: 1.25, fontWeight: 600, letterSpacing: '-0.016em'},
    h4: {fontSize: '1.125rem', lineHeight: 1.35, fontWeight: 600, letterSpacing: '-0.01em'},
    body1: {fontSize: '1.0625rem', lineHeight: 1.62, letterSpacing: '-0.005em'},
    body2: {fontSize: '0.9375rem', lineHeight: 1.6},
    button: {textTransform: 'none', fontWeight: 550, letterSpacing: 0},
    caption: {fontSize: '0.8125rem', lineHeight: 1.5, fontWeight: 500},
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        html: {WebkitFontSmoothing: 'antialiased', scrollBehavior: 'smooth'},
        body: {fontSynthesis: 'none'},
        'code, pre, kbd': {fontFamily: monoFamily},
        // Syntax colours live here rather than in the palette because they are four values
        // per scheme, not a palette role, and because a snippet then follows the scheme
        // without React re-rendering. See components/CodeBlock.tsx.
        ':root': {
          '--surface-wash': tokens.wash,
          '--surface-code': tokens.codeWash,
          '--code-keyword': tokens.lumen,
          '--code-string': tokens.lumenDeep,
          '--code-comment': tokens.slate,
          '--code-call': tokens.ink,
        },
        '[data-color-scheme="dark"]': {
          '--surface-wash': dark.wash,
          '--surface-code': dark.wash,
          '--code-keyword': dark.primary,
          '--code-string': '#90caf9',
          '--code-comment': dark.textSecondary,
          '--code-call': dark.text,
        },
        '@media (prefers-reduced-motion: reduce)': {
          html: {scrollBehavior: 'auto'},
          '*': {animationDuration: '0.01ms !important', transitionDuration: '0.01ms !important'},
        },
        // Keyboard focus must stay visible everywhere. This covers links; buttons need the MuiButtonBase override below, because ButtonBase sets
        // `outline: 0` on its root and wins the cascade against a bare :focus-visible
        // selector. Both read --focus-ring, which any inverted surface can reset.
        ':focus-visible': {outline: `2px solid ${FOCUS_RING}`, outlineOffset: 2},
      },
    },
    MuiButtonBase: {
      styleOverrides: {
        root: {'&:focus-visible': {outline: `2px solid ${FOCUS_RING}`, outlineOffset: 2}},
      },
    },
    MuiButton: {
      defaultProps: {disableElevation: true},
      styleOverrides: {
        root: {
          borderRadius: 6,
          paddingInline: 20,
          paddingBlock: 9,
          transition: 'box-shadow 200ms ease, background-color 200ms ease, border-color 200ms ease',
        },
        // Raised only where hover exists, in the scheme's own primary.
        contained: {
          '@media (hover: hover)': {
            '&:hover': {boxShadow: `0 8px 26px rgb(${PRIMARY} / 0.5)`},
          },
        },
        sizeLarge: {fontSize: '1rem', paddingInline: 26, paddingBlock: 12},
      },
    },
    MuiPaper: {defaultProps: {elevation: 0}},
    MuiLink: {defaultProps: {underline: 'none'}, styleOverrides: {root: {fontWeight: 500}}},
  },
})

export {monoFamily}
