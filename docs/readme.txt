# how to make sphinx documentation in an existing git repo, e.g. 
# /Users/Emily/Dropbox/pylanet

# step 1: install sphinx
conda install sphinx

# step 2: make documentation folder
cd /Users/Emily/Dropbox/pylanet
mkdir docs
cd ./docs

# step 3: initialize sphinx
sphinx-quickstart

# options to select during sphinx quickstart:
# > Separate source and build directories (y/n) [n]: y

# The project name will occur in several places in the built documentation.
# > Project name: pylanet
# > Author name(s): Zephyr Penoyre and Emily Sandford
# > Project release []: 
# 
# If the documents are to be written in a language other than English,
# you can select a language here by its language code. Sphinx will then
# translate text that it generates into that language.
# 
# For a list of supported codes, see
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-language.
# > Project language [en]: 

# Creating file ./source/conf.py.
# Creating file ./source/index.rst.
# Creating file ./Makefile.
# Creating file ./make.bat.

# Finished: An initial directory structure has been created.

# step 4: edit pylanet/docs/source/conf.py
# specifically: after import os, import sys, add:
# sys.path.insert(0, os.path.abspath('../..'))
# extensions = ['sphinx.ext.autodoc','sphinx.ext.coverage','sphinx.ext.napoleon']

# step 5: edit pylanet/docs/source/index.rst to tell it which sub-modules to index
# e.g. to index pylanet/model.py, add
# .. automodule:: pylanet.model
#    :members:

# step 6: build the docs. from within pylanet/docs, run
make html

