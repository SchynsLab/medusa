# Medusa: 4D face reconstruction and analysis

[![CI](https://github.com/medusa-4D/medusa/actions/workflows/tests.yaml/badge.svg)](https://github.com/medusa-4D/medusa/actions/workflows/tests.yaml)
![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/lukassnoek/420039a0fe8fb8c1170e0478cdcd0f26/raw/medusa_coverage_badge.json)
![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/lukassnoek/cb6da52c965ec24f136b74a1ebad1964/raw/medusa_interrogate_badge.json)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)

Medusa is Python toolbox for face image and video analysis. It offers tools for face
detection, alignment, rendering, and most importantly, *4D reconstruction*.
Using state-of-the-art 3D reconstruction models, Medusa can track and reconstruct faces
in videos (one 3D mesh per face, per frame) and thus provide a way to automatically
measure and quantify face movement as 4D signals. For an overview of the package and the
underlying methods, check out the video below.

<iframe width="560" height="315" src="https://www.youtube.com/embed/fnKfWwlrn6Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Documentation overview

On this website, you can find general information about Medusa (such as how to
[install](getting_started/installation) and [cite](getting_started/citation) it), as
well as details on Medusa's [command-line interface](api/cli) and
[Python interface](api/python).

## Tutorials

On this website you can also find several tutorials on Medusa's features:

::::{grid}
:gutter: 3

:::{grid-item-card} Quickstart
:img-background: ./images/quickstart_img.png
:link: ./getting_started/quickstart
:link-type: ref
:::

:::{grid-item-card} 4D reconstruction
Here's the second card.
:::

:::{grid-item-card} Data representation
Here's the third card.
:::

:::{grid-item-card} Rendering
Here's the third card.
:::

::::

A great way to get more familiar with the package is to check out the [quickstart](./getting_started/quickstart)!
