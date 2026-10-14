/**
 * All copy and link targets for the homepage, kept out of the components so wording can
 * be reviewed without reading JSX.
 *
 * The homepage is copied over the root of the Zensical build, so docs links are root
 * relative directory URLs. `homepage-serve` points them at the live site instead.
 */

export const DOCS = import.meta.env.VITE_DOCS_PATH ?? ''
export const GITHUB = 'https://github.com/holoviz/lumen'
const DISCOURSE = 'https://discourse.holoviz.org/c/lumen/'
const DISCORD = 'https://discord.com/invite/rb6gPXbdAr'

export const nav = [
  {label: 'Docs', href: `${DOCS}/overview/`},
  {label: 'Quick Start', href: `${DOCS}/quick_start/`},
  {label: 'Tutorials', href: `${DOCS}/examples/tutorials/`},
  {label: 'Gallery', href: `${DOCS}/examples/gallery/`},
  {label: 'API', href: `${DOCS}/reference/api/`},
]

export const getStarted = {label: 'Get started', href: `${DOCS}/quick_start/`}

export const social = [
  {label: 'GitHub', href: GITHUB},
  {label: 'Discord', href: DISCORD},
  {label: 'Discourse', href: DISCOURSE},
]

export const footerColumns = [
  {
    heading: 'Learn',
    links: [
      {label: 'Quick start', href: `${DOCS}/quick_start/`},
      {label: 'Installation', href: `${DOCS}/installation/`},
      {label: 'Getting started', href: `${DOCS}/getting_started/launching_lumen/`},
      {label: 'Tutorials', href: `${DOCS}/examples/tutorials/`},
    ],
  },
  {
    heading: 'Reference',
    links: [
      {label: 'Configuration', href: `${DOCS}/configuration/cli/`},
      {label: 'API reference', href: `${DOCS}/reference/api/`},
      {label: 'Gallery', href: `${DOCS}/examples/gallery/`},
      {label: 'Release notes', href: `${DOCS}/releases/`},
    ],
  },
  {
    heading: 'Community',
    links: [
      {label: 'Discourse', href: DISCOURSE},
      {label: 'Discord', href: DISCORD},
      {label: 'GitHub issues', href: `${GITHUB}/issues`},
      {label: 'Contributing', href: `${DOCS}/contributing/`},
    ],
  },
  {
    heading: 'HoloViz',
    links: [
      {label: 'Panel', href: 'https://panel.holoviz.org'},
      {label: 'hvPlot', href: 'https://hvplot.holoviz.org'},
      {label: 'HoloViews', href: 'https://holoviews.org'},
      {label: 'Param', href: 'https://param.holoviz.org'},
    ],
  },
]
