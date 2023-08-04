# How to cite Medusa and its dependencies

If you use Medusa in your research, please cite it as follows:

`````{tab-set}
````{tab-item} APA
Snoek, L., Jack, R., & Schyns, P. (2023, January 7). Dynamic face imaging: a novel analysis framework for 4D social face perception and expression. *IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG)*, 1-4. https://doi.org/10.1109/FG57933.2023.10042724.
````

````{tab-item} BibTeX
```bibtex
@inproceedings{snoek2023dynamic,
  title={Dynamic face imaging: a novel analysis framework for 4D social face perception and expression},
  author={Snoek, Lukas and Jack, Rachael E and Schyns, Philippe G},
  booktitle={IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG)},
  pages={1--4},
  year={2023},
  organization={IEEE}
}
```
````
`````

As Medusa relies on models from [Insightface]() for face detection, alignment, and
cropping, please cite the following two papers as well:

> Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). Retinaface: Single-shot multi-level face localisation in the wild. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 5203-5212).

> Guo, J., Deng, J., Lattas, A., & Zafeiriou, S. (2021). Sample and computation redistribution for efficient face detection. *arXiv preprint arXiv:2105.04714*.

## Citing reconstruction models

Please also cite the following, depending on the reconstruction model you use.

### Mediapipe

> Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019). Real-time facial surface geometry from monocular video on mobile GPUs. *arXiv preprint arXiv:1907.06724*

> Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). Mediapipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*.

### DECA / EMOCA
> Danecek, R., Black, M. J., & Bolkart, T. (2022). EMOCA: Emotion Driven Monocular
Face Capture and Animation. *arXiv preprint arXiv:2204.11312*.

> Feng, Y., Feng, H., Black, M. J., & Bolkart, T. (2021). Learning an animatable detailed 3D face model from in-the-wild images. *ACM Transactions on Graphics (TOG), 40*(4), 1-13.

### MICA

> Zielonka, W., Bolkart, T., & Thies, J. (2022, November). Towards metrical reconstruction of human faces. In *Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XIII* (pp. 250-269). Cham: Springer Nature Switzerland.

## Misc

If you use Medusa's `SCRFDetector`, please cite the following paper from the InsightFace team:

> Guo, J., Deng, J., Lattas, A., & Zafeiriou, S. (2021). Sample and computation redistribution for efficient face detection. *arXiv preprint arXiv:2105.04714.*

If you use Medusa's `AlignCropModel`, which is based on the landmark detection module of RetinaFace, please cite the following paper:

> Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). Retinaface: Single-shot multi-level face localisation in the wild. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 5203-5212).

## Website

This website (including documentation) was created using Jupyter Book {cite}`executable_books_community_2020_4539666`.
