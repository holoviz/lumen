import os

from setuptools import find_packages, setup

########## autover ##########

def get_setup_version(reponame):
    """Use autover to get up to date version."""
    # importing self into setup.py is unorthodox, but param has no
    # required dependencies outside of python
    from param.version import Version
    return Version.setup_version(os.path.dirname(__file__),reponame,archive_commit="$Format:%h$")


########## dependencies ##########

dependencies = [
    "numpy",
    "bokeh",
    "param >=1.9.0",
    "panel >=1.3.0",
    "pandas",
    "hvplot",
    "holoviews >=1.17.0",
    "packaging",
    "intake",
    "jinja2 >3.0"
]

extras_require = {
    'sql': [
        'duckdb',
        'intake-sql',
        'sqlalchemy <2',  # Don't work with pandas yet
    ],
    'tests': [
        'pytest',
        'flake8',
        'intake',
        'fastparquet',
        'msgpack-python',
        'toolz',
        'pre-commit',
        'matplotlib >=3.4',  # Ubuntu + Python 3.9 installs old version matplotlib (3.3.2)
        'pandas <2.2',
    ],
    'tests_ci' : [
        'pytest-github-actions-annotate-failures',
        'pytest-cov',
        'codecov',
    ],
    'doc': [
        'nbsite >=0.8.2',
    ]
}

extras_require['all'] = sorted(set(sum(extras_require.values(), [])))

########## metadata for setuptools ##########

setup_args = dict(
    name='lumen',
    version=get_setup_version("lumen"),
    description='A monitoring solution built on Panel.',
    long_description=open('README.md').read() if os.path.isfile('README.md') else 'Consult README.md',
    long_description_content_type="text/markdown",
    author="HoloViz",
    author_email="developers@holoviz.org",
    maintainer="HoloViz",
    maintainer_email="developers@holoviz.org",
    platforms=['Windows', 'Mac OS X', 'Linux'],
    license='BSD',
    url='https://github.com/holoviz/lumen',
    packages=find_packages(),
    provides=["lumen"],
    include_package_data = True,
    python_requires=">=3.9",
    install_requires=dependencies,
    extras_require=extras_require,
    tests_require=extras_require['tests'],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries"
    ],
    entry_points={
        'panel.io.rest': [
            "lumen = lumen.rest:lumen_rest_provider"
        ],
        'console_scripts': [
            'lumen = lumen.command:main'
        ]
    }
)

if __name__=="__main__":
    setup(**setup_args)
