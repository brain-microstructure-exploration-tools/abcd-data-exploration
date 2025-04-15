# abcd-data-exploration/prototype

Here live miscellaneous scripts and notebooks used for exploration of tabular ABCD Study data.

Notebooks that are `.py` files can be opened with Jupytext. They can be converted to `*.ipynb` files with
```bash
jupytext --to ipynb my_file.py
```

Scripts include:

- **[combineSegmentations.py](combineSegmentations.py)** - For each segment of a brain, combines registered individuals to produce an all-segments label map
- **[nilearn_exploration.py](nilearn_exploration.py)** - Built from an [example from nilearn](https://nilearn.github.io/stable/auto_examples/05_glm_second_level/plot_second_level_association_test.html)
- **[nilearn_prototype.py](nilearn_prototype.py)** - A full workflow that reads ABCD data, processes it, and displays outputs. Processing with nilearn is debugged.  Processing with numpy is supported, but not quite as complete.
- **[transformSegmentationsToTemplate.sh](transformSegmentationsToTemplate.sh)** - For each segment of a brain, registers images across individuals using the transforms computed for whole brain images
