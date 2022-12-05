# Clear any existing auto-generated documentation
rm -rf docs/_build
rm -rf docs/autoapi/_build
rm -rf docs/api/python/*
mkdir -p docs/_build/

# Build the API documentation with autoapi
sphinx-build -b html docs/autoapi docs/autoapi/_build

# Move the generated rst files to somewhere Jupyter book can find them
cp -r docs/autoapi/_build/_sources/autoapi/medusa/* docs/api/python/
rm -r docs/api/python/index.rst.txt

# Fix the weird rst.txt extension
for f in $(find docs/api/python -name \*.rst.txt); do
    mv $f ${f/.rst.txt/.rst}
done

# Finally build the entire docs
jupyter-book build docs/
rm -rf docs/api/python/*
