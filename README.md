# NanoFlavour

### Setup

```bash
cmsrel CMSSW_12_5_0
cd CMSSW_12_5_0/src
cmsenv

# get python3 compatible NanoAOD-tools
git clone git@github.com:hqucms/nanoAOD-tools.git PhysicsTools/NanoAODTools -b cmssw-py3

# get NanoFlavour (and switch to the desired branch)
git clone git@github.com:ttH-charm/NanoFlavour.git PhysicsTools/NanoFlavour

# compile
scram b -j8
```

### Test

Instructions to run the nanoAOD postprocessor can be found at [nanoAOD-tools](https://github.com/cms-nanoAOD/nanoAOD-tools#nanoaod-tools).

### Production

```bash
cd PhysicsTools/NanoFlavour/run
```

##### Make trees for heavy flavour tagging (bb/cc) or top/W data/MC comparison and scale factor measurement:

```bash
./runFlavTrees.py -o /path/to/output
(--sample-dir custom_samples -i /eos/cms/store/cmst3/group/hh/NanoAOD/20220722_NanoAODv9_pnWithTau/2018)
--channel [ZJets|WJets|TT1L|TT2L] --year [2016|2017|2018] -n 10
(--batch) (--type [data|mc|data,mc]) (--run-syst)
(--condor-extras '+AccountingGroup = "group_u_CMST3.all"')
```

To submit the created jobs (unless --batch option given):

condor_submit [jobdir]/[jobtypedir]/submit.cmd

To run locally for testing:

cd [jobdir]/[jobtypedir]/
./processor.py 1

Command line options:

- the preselection for each channel is coded in `runHRTTrees.py`
- use `--type` to select whether to run `data`, `mc` or both (`data,mc`)
- use `--run-syst` to make the systematic trees
- can run for multiple years together w/ e.g., `--year 2016,2017,2018`
- use `--sample-dir` to specify the directory containing the sample lists. Currently we maintain two sets of sample lists: the default one is under [samples](run/samples) which is used for running over official NanoAOD datasets remotely, and the other one is [custom_samples](run/custom_samples) which is used for running over privately produced NanoAOD datasets locally.
  - To run over the private produced samples, ones needs to add `--sample-dir custom_samples` to the command line, and add `-i /path/to/files` to set the path to the custom NanoAOD samples.
  - To run over (remote) official NanoAODs, remove `-i` and **consider adding `--prefetch`** to copy files first before running to increase the processing speed. **Also reduce to `-n 1` or `-n 2`**, i.e., 1 or 2 files per job, as the official NanoAODs have many events per file.
- add the `--batch` option to submit jobs to condor automatically without confirmation
- **[NEW]** use `--condor-extras` to pass extra options to condor job description file

More options of `runPostProcessing.py` or `runFlavTrees.py` (a wrapper of `runPostProcessing.py`) can be found with `python runPostProcessing.py -h` or `python runFlavTrees.py -h`, e.g.,

- To resubmit failed jobs, run the same command but add `--resubmit`.

- To add cross section weights and merge output trees according to the config file, run the same command but add `--post`. The cross section file to use can be set with the `-w` option.

### Checklist when updating to new data-taking years / production campaigns

- [ ] triggers
- [ ] lumi values
- [ ] golden JSON
- [ ] PU rewgt
- [ ] lepton ID/ISO
- [ ] b-tag WP
- [ ] JEC/JER
- [ ] MET filters
- [ ] MET recipes (if any)
- [ ] `keep_and_drop` files
- [ ] samples (check also those in PRODUCTION status)
