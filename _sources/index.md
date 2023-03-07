# Medusa: 4D face reconstruction and analysis

[![CI](https://github.com/medusa-4D/medusa/actions/workflows/tests.yaml/badge.svg)](https://github.com/medusa-4D/medusa/actions/workflows/tests.yaml)
[![CI](https://github.com/medusa-4D/medusa/actions/workflows/docs.yaml/badge.svg)](https://medusa.lukas-snoek.com/medusa)
![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/lukassnoek/420039a0fe8fb8c1170e0478cdcd0f26/raw/medusa_coverage_badge.json)
![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/lukassnoek/cb6da52c965ec24f136b74a1ebad1964/raw/medusa_interrogate_badge.json)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)

Medusa is Python toolbox for face image and video analysis. It offers tools for face
detection, alignment, rendering, and most importantly, *4D reconstruction*.
Using state-of-the-art 3D reconstruction models, Medusa can track and reconstruct faces
in videos (one 3D mesh per face, per frame) and thus provide a way to automatically
measure and quantify face movement as 4D signals.

In Medusa, 4D reconstruction data is represented as a series of 3D meshes. Each mesh
describes the face shape at a particular frame in the video, and the changes in the
meshes over time thus decribe facial *movement* (including expression) quantitatively
and dynamically. Medusa makes relatively few assumptions as to how you want to (further)
analyze the face and just returns the raw set of vertices. For some ideas on
how to analyze such data, check out the [analysis tutorials](tutorials/analysis) (WIP).

## Documentation overview

On this website, you can find general information about Medusa (such as how to
[install](getting_started/installation) and [cite](getting_started/citation) it), as
well as several tutorials and details on Medusa's [command-line interface](api/cli) and
[Python interface](api/python).

A great way to get more familiar with the package is to check out the [quickstart](getting_started/quickstart)!
