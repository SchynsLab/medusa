import os
from setuptools import setup, find_packages

PACKAGES = find_packages()

# Get version and release info, which is all stored in shablona/version.py
ver_file = os.path.join('gmfx', 'version.py')

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
        'opencv-python',
        'pandas',
        'numpy',
        'imageio',
        'tqdm',
        'matplotlib',
        'scikit-image',
        'face_alignment',
        'pyyaml',
        'h5py',
        'tables',  # to use hdf5 with pandas
        'trimesh',
        'nilearn',
        'imageio-ffmpeg',
        'chumpy'  # for FLAME model .. really necessary?
    ],
    entry_points={
        'console_scripts': [
            'gmfx_videorecon = gmfx.cli:videorecon_cmd',
            'gmfx_align = gmfx.cli:align_cmd',
            'gmfx_resample = gmfx.cli:resample_cmd',
            'gmfx_filter = gmfx.cli:filter_cmd',
            'gmfx_preproc = gmfx.cli:preproc_cmd',
            'gmfx_create_cfg = gmfx.cli:create_cfg_cmd'
        ]
    }
)

if __name__ == '__main__':
    setup(**opts)
