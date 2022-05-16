import os
from setuptools import setup, find_packages

PACKAGES = find_packages()

# Get version and release info, which is all stored in shablona/version.py
ver_file = os.path.join('medusa', 'version.py')

with open(ver_file) as f:
    exec(f.read())

# Long description will go up on the pypi page
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()


opts = dict(name=NAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    version=VERSION,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    install_requires=[
        'click',
        'pandas',
        'numpy',
        'imageio',
        'imageio-ffmpeg',
        'opencv-python',
        'tqdm',
        'matplotlib',
        'scikit-image',
        'face_alignment',
        'pyyaml',
        'h5py',
        'tables',  # to use hdf5 with pandas
        'trimesh',
        'chumpy'  # for FLAME model .. really necessary?
    ],
    entry_points={
        'console_scripts': [
            'medusa_videorecon = medusa.cli:videorecon_cmd',
            'medusa_align = medusa.cli:align_cmd',
            'medusa_resample = medusa.cli:resample_cmd',
            'medusa_filter = medusa.cli:filter_cmd',
            'medusa_preproc = medusa.cli:preproc_cmd',
            'medusa_epoch = medusa.cli:epoch_cmd',
            'medusa_videorender = medusa.cli:videorender_cmd'
        ]
    }
)

if __name__ == '__main__':
    setup(**opts)
