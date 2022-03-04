# Run all the tests

pytest


# Code coverage

Requires: pip3 install coverage

coverage erase
coverage run --source pvml -m pytest
coverage report -m
