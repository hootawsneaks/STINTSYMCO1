# Binary Classification (Fractured / Non-Fractured) on X-Ray images

**By**: Kean Carvin, Gideon Chua, Jean-Luc Gaffud, Bea Uy

**Dataset**: Abedeen et al. (2023). FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs [https://doi.org/10.1038/s41597-023-02432-4](https://doi.org/10.1038/s41597-023-02432-4)

**Motivation**: Identification of Fractures with both high recall and performance via machine learning can allow healthcare workers to focus on other important tasks.

**Goal**: By the end of the project, our goal is to create and compare supervised learning algorithms for Fracture identification.

## Setup 

> This project requires [uv](https://docs.astral.sh/uv/getting-started/installation/) as its package manager.

1. Install the [FracAtlas Dataset](https://doi.org/10.6084/m9.figshare.22363012) and place it on the root directory inside the project.
2. Clone the repository.
3. Run `uv sync` which should catch your environment up.
4. Run it as a kernel on Jupyter notebook.

## Declaration of AI usage
The group used Microsoft Copilot and Claude Sonnet 4.6 for:
- Brainstorming ideas on what models can be implemented for the dataset
- Brainstorming possible libraries (such as albumentations) and algorithms for certain problems
