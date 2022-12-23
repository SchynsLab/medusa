# # Unit tests
coverage run --source=medusa/ -m pytest --disable-warnings tests/test_transforms.py

# # Doctests
# pytest --disable-warnings --doctest-modules --exitfirst medusa/

# # Clean up stuff from doctests
# for f in example_vid_recon.mp4 example_vid_recon.mp4; do
#     if [ -f $f ]; then
#         rm $f
#     fi
# done

# # Run notebooks
# pytest --nbval docs/*/*ipynb
