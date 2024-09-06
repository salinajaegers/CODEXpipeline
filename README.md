# Pipelining Mulitvariate CODEX

This is a pipeline made for CODEX (https://github.com/pertzlab/CODEX) using snakemake. The Pipeline is using the CNN-training, the prototype selection, and the motif mining from CODEX. A PCA has been added as well as the option to have an interactive exploration of the PCA results in a notebook separate from the pipeline. The motif analysis part of CODEX which is done in R has not yet been integrated into the pipeline and needs to be set up locally. 
CODEX has been extended to include a model that is able to process multivariate time series. It was previously restricted to two measurements per ID.

## Installing CODEX
Tthe environment for CODEX is created using a singularity image or CONDA. 

To create a singularity environment run `singularity build sing_CODEX_CPU_pipeline_motif.sif sing_CODEX_CPU_pipeline_motif.def` to creat the image. Currently the image is only woking with CPU. Alternatively, a GPU-compatible environment can be created in CONDA by running `conda env create -f CODEX-pipeline-conda.yml`. Both environement files can be found in the `conda` folder. 

## Data preparation
The input to the pipeline is similar to what is required from CODEX. The class definition file and the dataset file need to be the same format as described in https://github.com/pertzlab/CODEX, except that interpolation is not necessary in the dataset as it is done within the pipeline. The set with the allocatement of training, validation, and test to the IDs is not required as input, it will be created with the percentages definined in the pipeline configurations. The data does not need to be zipped either as an archive will be created after creating the id allocation set. It does however need to be in a separate directory, which Snakemake can be run in. 

## Running CODEX
This is a short description of how to use the pipeline for CODEX:

1. Create two files according to original description:
    - `dataset.csv`: this is the file that contains the time-series. The data are organized in 
    wide format (series on rows, measurements on columns). It must contain 2 columns: ID and class 
    (the column names are flexible but these names will be automatically recognized without 
    further input from the user). The ID should be unique to each series. The classes should be 
    dummy-coded with integer (e.g. if you have 3 classes named A, B and C, this should be encoded 
    as 0, 1, or 2 in this column). The rest of the columns should contain the actual measurements. 
    The column names must follow a precise convention. They are of the form XXX_YYY, where XXX 
    represent the measurement name and YYY the time of measurement (e.g. ERK_12, means ERK value 
    at time point 12). Shall you have multivariate series, just append new columns while 
    respecting the naming convention. For example, for a dataset of 3 time points where you follow 
    both ERK and AKT in single cells the column names should be: 
        `ID, class, ERK_1, ERK_2, ERK_3, AKT_1, AKT_2, AKT_3`
    - `classes.csv`: this file holds the correspondence between the dummy-coded classes and their 
    actual names. It is a small file with 2 columns: class and class_name. The former holds the 
    same dummy-coded variables as in dataset.csv; while the second holds the human-readable 
    version of it. Once again please try to stick to these default column names so you do not have to 
    pass them at every DataProcesser call.
    - In the original version it is also required to make an training/validation split file but that is 
    integrated in the pipeline and therefore not necessary here


2. Put both the dataset.csv and the classes.csv file in the same folder. 
    Additionally, a configuration file is provided (`config.yaml`) which should go into the same 
    directory. That is where the many parameters for CODEX are set. 
    Your directory for CODEX should now look like this:
    - `directory`:
        - `dataset.csv`
        - `classes.csv`
        - `config.yaml`

3. Now set the parameters in the config.yaml file. The important parameters to set will be marked with #######
    before their explainations and everything else can be left as is. All parameters should be 
    explained in detail in the CODEX notebooks, in case anyone wants to try different things or get more clarification on the backgroud of the model.

4. To run the pipeline activate the conda environment by typing 'conda activate codex' into the terminal. 
    In case the environment isn't setup yet, use the environemnt yml file to create one: 
    `conda env create -f CODEX-pipeline-conda.yml`
    
5. Adjust the working directory $WD variable in the slurm job file PIPELINE.sh by setting it to the path of the directory
    with the data and configuration file and run it. Depending on 
    how large your dataset is you might also want to adjust the slurm job resources. 
    For 150'000 cells measured, we used 20h, with 12 cpus where each had 30gb of memory. <br />
    `#SBATCH --cpus-per-task=12` <br />
    `#SBATCH --mem-per-cpu=30GB` <br />
    `#SBATCH --time=20:00:00` <br />
    Consider running the pipeline on GPU instead, it would make it much faster and use less resources.

    Alternatively, the pipeline can be run locally from the commandline with `snakemake -s ./source/Snakefile.txt -d path/to/working/directory --configfile ./source/config.yaml`.

    NOTE: It is only possible to run one analysis at a time per working directory, as snakemake does not allow for more.

## Interactive PCA
To explore the feature space in the form of a PCA to gain more insight into what differentiates the classes, we provide an interactive PCA plot. It is best to run this part locally, instead of on a cluster. To make it more accessible for local usage, the PCA is calculated as part of the pipeline and the results can then be referenced in the notebook. This way only the plots themselves have to be created locally. The plot can then be explored in more detail by clicking on the points within the plot itself. A tab should open in the browser where the class probabilities and the time series are shown. Each point that is clicked on with its details will be saved as an .html file in the repository where the interactive PCA is run in. So there is always the possibility to go back to previous explored regions.

To run the interactive PCA it is recommended to use jupyter notebook instead of VSCode. 

1. In the terminal type go to the folder where the interactive Notebook is stored. Then enter
        `jupyter notebook`

    This will start up a jupyter notebook on your browser.

2. In the notebook adjust the paths to the datset.csv and the PCA results from the CODEX analysis.

3. Shift + Enter to execute the notebook

4. On the bottom the clickable PCA should now be displayed. Clicking on any point opens up a tab and 
    saves a .html file with the information about that point in the interactivePCA directory. 
    (Note: Only the PCA in the notebook is interactive, not the new ones opened in the tabs.)

