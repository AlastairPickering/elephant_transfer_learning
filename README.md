# A scalable transfer learning workflow for extracting biological and behavioural insights from forest elephant vocalizations

This repository provides a workflow for extracting acoustic features from elephant vocalisations using transfer learning and unsupervised dimensionality reduction techniques.
It includes the necessary code and utility scripts to replicate the analysis. It also includes a small sample of the original audio files. The full dataset is available upon request.

## Contents

- **`1_feature_extraction_notebook.ipynb`:** This notebook extracts acoustic features of annotated elephant vocalisations using four different pre-trained CNN models.
- **`2_call_types_notebook.ipynb`:** This notebook uses the extracted acoustic features to analyse how well they retain structural information needed to distinguish different elephant call types.
- **`3_behaviour_demography_notebook.ipynb`:** This notebook uses the extracted acoustic features to assess how well they retain behavioural and demographic information.
- **`audio_dir`:** Contains the raw audio recordings of elephant vocalisations.
- **`data`:** Contains tables with metadata associated to the recordings and vocalisations (see description below for more details).
- **`elephant_scripts`:** Contains pre-defined functions used to process each stage of the workflow. Note you will need to download the BirdNET model: https://github.com/birdnet-team/BirdNET-Analyzer
- **`outputs`:** Contains pre-computed intermediate outputs to avoid computation-heavy steps.

## Dataset Description

This repository analyses a dataset of African forest elephant vocalisations recorded by the Elephant Listening Project.
These recordings were collected in Dzanga-Bai clearing, Central African Republic, between September 2018 and April 2019.
The dataset includes:

1. Audio Recordings: Raw audio files of elephant vocalisations (a sample of which is located in the audio_dir folder).
2. `data/elephant_vocalisations.csv`: A table of 1254 annotated vocalisations, each with:
   - Vocalisation ID
   - Start and end time
   - Frequency range
   - Call type (roar, rumble, or trumpet)
   - Corresponding audio file name
3. `data/elephant_contextual_observations.csv`: A table with 359 entries, each describing the context of an audio recording of a rumble vocalisation.
   This includes information about the vocalising elephants (e.g., age, sex, behaviour) and the overall recording context (e.g., signs of distress).
   Each audio file may contain multiple vocalisations, but all share the same contextual information.

## How to Use

1. **Install Dependencies:**
   - Create a virtual environment (recommended).
   - Install the required packages listed in `requirements.txt` or the `pyproject.toml`.
2. **Run the Notebooks:**
   - Start with the notebook `1_feature_extraction_notebook.ipynb` to learn how acoustic features of vocalisations are automatically extracted using a pre-trained CNN.
   - Proceed to `2_call_type_notebook.ipynb`, `3_behaviour_demography_notebook.ipynb` to explore the analysis of call types and the influence of behaviour on vocalisations.

### Installing Dependencies

The (arguably) easiest way to create a Python environment with all required dependencies is using [uv](https://docs.astral.sh/uv/).

1. **Install uv:**

   - For MacOS/Linux:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - For Windows: `powershell powershell -c "irm https://astral.sh/uv/install.ps1 | iex" ` You can also follow the instructions in the [documentation](https://docs.astral.sh/uv/).

2. **Create a virtual environment and install dependencies:**

   ```bash
   uv sync
   ```

3. **(Optional) Setup a Jupyter notebook kernel:**

   ```bash
   uv run python -m ipykernel install --user --name elephant
   ```

   Change the name of the kernel to your preference.
