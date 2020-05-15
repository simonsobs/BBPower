BBPower - the C_ell-based pipeline for BB
-----------------------------------------

This repo hosts a pipeline that carries out a maps-to-params analysis of multi-frequency polarization data to constrain primordial B-modes using a power-spectrum-based component separation scheme. The pipeline is built following the BBPipe framework. [BBPipe](https://github.com/simonsobs/BBPipe) is a pipeline constructor used to connect different pipeline stages in terms of their outputs and required inputs, and it's one of BBPower's dependencies.

### Dependencies
You should install the following non-standard python packages in order to use BBPower:
- [BBPipe](https://github.com/simonsobs/BBPipe)
- [sacc](https://pypi.org/project/sacc/)
- [emcee](https://pypi.org/project/emcee/)

### Using the code
First of all, have a look at the BBPipe documentation to get a broad idea of how the pipeline structure works. Then, have a look at one of the test suites in the `test` directory. For instance, a quick pipeline that takes calculated power spectra, runs an MCMC on them and creates a bunch of plots, is contained in the following files:
- [test/test_sampling.yml](test/test_sampling.yml) describes the raw inputs of the pipeline (in this case, multi-frequency power spectra) and its stages (in this case, two stages, a likelihood sampling stage and a plotting stage).
- [test/test_config_sampling.yml](test/test_config_sampling.yml) describes the configuration options for the different stages. This includes, for instance, the cosmological and foreground model, parameter priors, sampler options etc. All the possible options are thoroughly described there.
- [test/run_sampling_test.sh](test/run_sampling_test.sh) contains the commands that would be needed to run this pipeline. The commands of the form `python -m bbpower ...` are output by `BBPipe` if you just run `bbpipe ./test/test_sampling.yml --dry-run`.

### Credits and questions
Get in touch with Max Abitbol (mabitbol), David Alonso (damonge) or anyone else in the SO BB AWG if you have questions or queries about the code.

