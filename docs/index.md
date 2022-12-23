# Medusa: 4D face reconstruction and analysis

[![CI](https://github.com/medusa-4D/medusa/actions/workflows/ci.yaml/badge.svg)](https://github.com/medusa-4D/medusa/actions/workflows/ci.yaml)
[![CI](https://github.com/medusa-4D/medusa/actions/workflows/docs.yaml/badge.svg)](https://medusa.lukas-snoek.com/medusa)
![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/lukassnoek/420039a0fe8fb8c1170e0478cdcd0f26/raw/medusa_coverage_badge.json)
![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/lukassnoek/cb6da52c965ec24f136b74a1ebad1964/raw/medusa_iterrogate_badge.json)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)

Medusa is a Python toolbox to perform 4D face reconstruction and analysis. You can use it
to reconstruct a series of 3D meshes of (moving) faces from video files: one 3D mesh for
each frame of the video (resulting in a "4D" representation of facial movement). In
addition to functionality to reconstruct faces, Medusa also contains functionality to
preprocess and analyze the resulting 4D reconstructions.

## When (not) to use Medusa?

More specifically, Medusa allows you to reconstruct, preprocess, and analyze
frame-by-frame time series of 3D faces from videos. The data that Medusa outputs is
basically a set of 3D points ("vertices"), which together represent face shape,
that move over time. Medusa then processes these points in a similar way that fMRI or
EEG/MEG software processes voxels or sensors, but instead of representing "brain activity",
it represents face movement! Medusa makes relatively few assumptions as to how you want
to (further) analyze the face and just returns the raw set of vertices. For some ideas on
how to analyze such data, check out the [analysis tutorials](tutorials/analysis).

## Documentation overview

On this website, you can find general information about Medusa (such as how to [install](getting_started/installation)
and [cite](getting_started/citation) it), as well as several tutorials
and details on Medusa's [command-line interface](api/cli) and [Python interface](api/python).

A great way to get more familiar with the package is to check out the [quickstart](getting_started/quickstart)!
