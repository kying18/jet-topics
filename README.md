# Jet Topic Modeling and Substructure Extraction

This code is designed to use the Markov Chain Monte Carlo (MCMC) emcee to calculate jet topics in proton-proton and heavy-ion collisions. This repo in particular is a refactored and modified version of the original by Jasmine Brewer and Andrew P. Turner with conceptual oversight from Jesse Thaler, located at https://github.com/jasminebrewer/jet-topics-from-MCMC. The topics extraction procedure is based on https://arxiv.org/abs/2008.08596. The topics' jet observable substructure extraction is based on a work in progress.

[Code Overview](## Code Overview)
[Data](## Samples)
[Run the Code](## How to Run the Code)
[Outputs](## Outputs)

## Code Overview
This repository provides the code for the topic modeling algorithm and subsequent substructure observable extraction. The whole pipeline of data can be expressed in the following image. This repo corresponds to the pink box:
![Pipeline leading up to topic modeling algorithm](https://github.com/kying18/jet-topics/blob/main/readme-imgs/pipeline.png?raw=true)

The repo is divided into three components: a `Data` object (located at `data.py`), a `Plotter` object (located at `plotter.py`), and a `Model` object (located at `model.py`). Below is a diagram representing how the components interact with one another:
![Components of this repo](https://github.com/kying18/jet-topics/blob/main/readme-imgs/model.png?raw=true)

The `Data` object is responsible for data ingestion from the `{system}.csv` file, parsing the file to create `Histogram` namedtuples. These `Histograms` keep track of various numpy arrays representing the x axis, unnormalized histogram, unnormalized histogram error, normalized histogram, normalized histogram error, and total entries in the histogram (int).

The `Plotter` object is responsible for creating all the plots throughout the process of running the algorithm.

The `Model` object is responsible for fitting the data using least squares and MCMC, extracting the kappa values, and calculating the extracted topics and subsequent topic substructures.

The three components interact with one another to produce results in `run.py`.

## Samples

Along with the code, we also provide sample dijet and photon+jet histograms in proton-proton and heavy-ion collisions which can be used as an example to run the code. The sample histograms are provided in the `data` folder. There is currently one file `pt80100.csv`, representing the PYQUEN data between `80<pT<100 GeV`. To run the code in its current form, the csv files containing the sample histograms should be saved in the the `data` directory.

For each sample, there should be 6 lines contained in the input sample in the following format:
`{sample_label},0,0,1,2,4,...` (histogram values)
`{sample_label}_error,0,0,1,1.41,2,...` (histogram error values)
`{sample_label}_quark_truth,0,0,1,2,4,...` (quark truth histogram values)
`{sample_label}_quark_truth_error,0,0,1,1.41,2,...` (quark truth histogram error values)
`{sample_label}_gluon_truth,0,0,1,2,4,...` (gluon truth histogram values)
`{sample_label}_gluon_truth_error,0,0,1,1.41,2,...` (gluon truth histogram error values)

## How to Run the Code

The syntax to run the code is

`./run.py system sample_1 sample_2 nwalkers nsamples burn_in nkappa x_min x_max xlabel`

Parameters:

`system` - string representing file name of the data file, located under `data` the data folder (not including `.csv`)

`sample_1`/`sample_2` - strings representing the labels of the inputs (assumed to correspond to photon jet sample and dijet sample); should be the prefix of the histogram values in the data file (see example below)

`nwalkers` - number of walkers for the MCMC

`nsamples` - number of samples taken by each walker in the MCMC

`burn_in` - number of samples after which the MCMC is thought to have converged to the posterior (it is required that `burn_in < nsamples`)

`nkappa` - number of samples from the posterior at which to sample kappa (it is required that `nkappa < (nsamples - burn_in) * nwalkers`)

`x_min`/`x_max` - absolute minimum or maximum index of the input histogram from which to ingest data (note: the code will automatically process the histogram such that we throw away unwanted 0s on either tail of the distribution... let's call these min_cut and max_cut... if x_min > min_cut or x_max < max_cut, then x_min and x_max will take priority, otherwise min_cut and max_cut will dictate the cuts)

`xlabel` - string representing label of input observable (x-axis)

Example parameter sets:

An example parameter set to get qualitative results with reasonable computational resources is

`./run.py pt80100 pp80_photonjet pp80 100 5000 4000 1000 0 100 "Constituent Multiplicity"`

## Outputs

The output of the code is a sequence of plots, saved in the `plots/{system}` directory, and a csv file containing the extracted values of kappa, saved in the `kappas/settings_*.csv` file.

`*input_unnormalized.png` - shows the unnormalized input histograms.

`*lstsq_fit.png` - shows the input histograms along with the least-squares fit obtained by the code. The MCMC walkers are started in a gaussian ball around these least squares fit parameters, so it is important that these fits are good. `LEAST_SQUARES_TRY_TIMES` in `config.py` specifies how many times to attempt a least-squares fit and the best one is kept; increasing trytimes may improve the fit if it is not converging.

`*mcmc_samples.png` - shows the parameters of the fit obtained by the MCMC walkers as a function of the time step. The vertical blue line is the value of burn_in specified as an input to the function. It is important that the walkers have converged by the burn_in time or results may be biased.

`*kappas.png` - shows the ratios pdf1/pdf2 and pdf2/pdf1 for the fits extracted from the MCMC (red) compared to the input histograms (black). The extracted values and locations of kappa_ab and kappa_ba are shown as blue dots.

`*topics.png` - shows the extracted topics (colored bands) compared to sample distributions of photon+quark, photon+gluon, dijet+quark, and dijet+gluon.

`*jet-\*.png` - shows the extracted topics' jet observables (currently supports jet shape, jet fragmentation, jet mass, and jet splitting fraction) as determined by the extracted kappas from the MCMC fits

`kappas/settings_*.csv` - a csv file containing the kappas extracted from this run of the MCMC. Along with the input distributions, these kappas can be used to reproduce the topics plots and the jet observables plots.

See https://emcee.readthedocs.io/en/stable/tutorials/line/ for a valuable tutorial on fitting with MCMC.