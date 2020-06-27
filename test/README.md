# Run all the tests

python3 -m unittest


# Code coverage

Requires: pip3 install coverage

coverage erase
coverage run --source pvml -m unittest
coverage report -m
