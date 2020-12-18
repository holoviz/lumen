import os

from setuptools import setup, find_packages

########## autover ##########

def get_setup_version(reponame):
    """Use autover to get up to date version."""
    # importing self into setup.py is unorthodox, but param has no
    # required dependencies outside of python
    from param.version import Version
    return Version.setup_version(os.path.dirname(__file__),reponame,archive_commit="$Format:%h$")


########## dependencies ##########

extras_require = {
    'tests': [
        'pytest',
        'flake8',
        'intake'
    ],
    'doc': [
        'sphinx',
        'pydata-sphinx-theme',
        'myst-parser',
        'nbsite'
    ]
}

extras_require['all'] = sorted(set(sum(extras_require.values(), [])))

########## metadata for setuptools ##########

setup_args = dict(
    name='lumen',
    version=get_setup_version("lumen"),
    description='A monitoring solution built on Panel.',
    long_description=open('README.rst').read() if os.path.isfile('README.rst') else 'Consult README.rst',
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
    python_requires=">=3.6",
    install_requires=["panel", "pandas"],
    extras_require=extras_require,
    tests_require=extras_require['tests'],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
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
