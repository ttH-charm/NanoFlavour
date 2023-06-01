#!/usr/bin/env python3
from __future__ import print_function

import os
import copy
import time

from runPostProcessing import get_arg_parser, run, tar_cmssw
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

zjets_cfgname  = 'zjets_cfg.json'
default_config = {'ele_scale': None,
                  'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': 'RegroupedV2_',
                  'jer': 'nominal', 'jmr': None, 'met_unclustered': None, 'applyHEMUnc': False,
                  'smearMET': False}

# _jes_uncertainty_sources = [
#     'AbsoluteMPFBias', 'AbsoluteScale', 'AbsoluteStat', 'FlavorQCD', 'Fragmentation', 'PileUpDataMC', 'PileUpPtBB',
#     'PileUpPtEC1', 'PileUpPtEC2', 'PileUpPtHF', 'PileUpPtRef', 'RelativeBal', 'RelativeFSR', 'RelativeJEREC1',
#     'RelativeJEREC2', 'RelativeJERHF', 'RelativePtBB', 'RelativePtEC1', 'RelativePtEC2', 'RelativePtHF',
#     'RelativeStatEC', 'RelativeStatFSR', 'RelativeStatHF', 'SinglePionECAL', 'SinglePionHCAL', 'TimePtEta',
# ]
#
# jes_uncertainty_sources = {
#     '2016': _jes_uncertainty_sources + ['RelativeSample'],
#     '2017': _jes_uncertainty_sources + ['RelativeSample'],
#     '2018': _jes_uncertainty_sources,
# }

jes_uncertainty_sources = {
    '2016': ['Absolute', 'Absolute_2016', 'BBEC1', 'BBEC1_2016', 'EC2', 'EC2_2016', 'FlavorQCD', 'HF', 'HF_2016', 'RelativeBal', 'RelativeSample_2016'],
    '2017': ['Absolute', 'Absolute_2017', 'BBEC1', 'BBEC1_2017', 'EC2', 'EC2_2017', 'FlavorQCD', 'HF', 'HF_2017', 'RelativeBal', 'RelativeSample_2017'],
    '2018': ['Absolute', 'Absolute_2018', 'BBEC1', 'BBEC1_2018', 'EC2', 'EC2_2018', 'FlavorQCD', 'HF', 'HF_2018', 'RelativeBal', 'RelativeSample_2018'],
}

golden_json = {
    '2015': 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
    '2016': 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
    '2017': 'Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt',
    '2018': 'Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt',
}


def _base_cut(year):
    # FIXME: remember to update this whenever the selections change in zjetsTreeProducer.py
    # FIXME: why not using ``Electron_mvaFall17V2Iso_WP90'' for 2L? ~40% gain in signal eff.
    cut_dict = {
        'ele_cut': 'Electron_pt>15 && abs(Electron_eta)<2.4 && Electron_mvaFall17V2Iso_WP90',
        'mu_cut': 'Muon_pt>15 && abs(Muon_eta)<2.4 && Muon_tightId && Muon_pfRelIso04_all<0.25',
        'jet_count': 'Sum$(Jet_pt>15 && abs(Jet_eta)<2.4 && (Jet_jetId & 4))',
    }
    basesels = {
      'zjets': '(Sum$({ele_cut}) + Sum$({mu_cut})) == 2 && '
               '(Sum$(Electron_pt>25 && {ele_cut}) + Sum$(Muon_pt>25 && {mu_cut})) >= 1 && '
               '{jet_count}<=2',
    }
    cut = basesels['zjets'].format(**cut_dict)
    return cut


def _process(args):
    year = args.year
    default_config['year'] = year

    basename = os.path.basename(args.outputdir) + '_' + year
    args.outputdir = os.path.join(os.path.dirname(args.outputdir), basename, args.type)
    args.jobdir = os.path.join('jobs_%s' % basename, args.type)
    args.datasets = '%s/%s/%s_%s.yaml' % (args.sample_dir, year, 'ZJets', args.type)
    args.cut = _base_cut(year)

    # New
    args.branchsel_in  = "keep_and_drop_input_zjets.txt"
    args.branchsel_out = "keep_and_drop_output_zjets.txt"

    args.imports = [('PhysicsTools.NanoFlavour.producers.zjetsTreeProducer', 'zjetsTreeFromConfig')]
    if args.type != 'data':
        args.imports.extend([
            # ('PhysicsTools.NanoTTH.producers.leptonSFProducerV2',
            # 'electronSF_{year}_{chn},muonSF_{year}_{chn}'.format(year=year, chn=channel)),
            # ('PhysicsTools.NanoTTH.producers.puJetIdSFProducer', 'puJetIdSF_' + year),
            ('PhysicsTools.NanoFlavour.producers.topPtWeightProducer', 'topPtWeight'),
            ('PhysicsTools.NanoAODTools.postprocessing.modules.common.puWeightProducer',
             'puWeight_UL2016' if year == '2015' else 'puWeight_UL%s' % year),
        ])

    # data, or just nominal MC
    if args.type == 'data' or not args.zjets_do_syst:
        args.cut = _base_cut(year)
        cfg = copy.deepcopy(default_config)
        if args.type == 'data':
            args.extra_transfer = os.path.expandvars(
                '$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data/JSON/%s' % golden_json[year])
            args.json = golden_json[year]
            cfg['jes'] = None
            cfg['jer'] = None
            cfg['jmr'] = None
            cfg['met_unclustered'] = None
        run(args, configs={zjets_cfgname: cfg})
        return

    # MC for syst.
    if args.type == 'mc' and args.zjets_do_syst:

        # nominal w/ PDF/Scale weights
        logging.info('Start making nominal trees with PDF/scale weights...')
        syst_name = 'LHEWeight'
        opts = copy.deepcopy(args)
        cfg = copy.deepcopy(default_config)
        opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
        opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
        opts.branchsel_out = 'keep_and_drop_output_zjets_LHEweights.txt'
        run(opts, configs={zjets_cfgname: cfg})

        # electron energy scale up/down
        # if channel != '0L':
        #     for variation in ['up', 'down']:
        #         syst_name = 'egamma_%s' % variation
        #         logging.info('Start making %s trees...' % syst_name)
        #         opts = copy.deepcopy(args)
        #         cfg = copy.deepcopy(default_config)
        #         cfg['ele_scale'] = variation
        #         opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
        #         opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
        #         run(opts, configs={tth_cfgname: cfg})

        # JMR up/down
        # for variation in ['up', 'down']:
        #     syst_name = 'jmr_%s' % variation
        #     logging.info('Start making %s trees...' % syst_name)
        #     opts = copy.deepcopy(args)
        #     cfg = copy.deepcopy(default_config)
        #     cfg['jmr'] = variation
        #     opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
        #     opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
        #     run(opts, configs={tth_cfgname: cfg})

        # JER up/down
        for variation in ['up', 'down']:
            syst_name = 'jer_%s' % variation
            logging.info('Start making %s trees...' % syst_name)
            opts = copy.deepcopy(args)
            cfg = copy.deepcopy(default_config)
            cfg['jer'] = variation
            opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
            opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
            run(opts, configs={zjets_cfgname: cfg})

        # MET unclustEn up/down
        #if channel != '2L':
        #    for variation in ['up', 'down']:
        #        syst_name = 'met_%s' % variation
        #        logging.info('Start making %s trees...' % syst_name)
        #        opts = copy.deepcopy(args)
        #        cfg = copy.deepcopy(default_config)
        #        cfg['met_unclustered'] = variation
        #        opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
        #        opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
        #        run(opts, configs={zjets_cfgname: cfg})

        # JES sources
        for source in jes_uncertainty_sources[year]:
            for variation in ['up', 'down']:
                syst_name = 'jes_%s_%s' % (source, variation)
                logging.info('Start making %s trees...' % syst_name)
                opts = copy.deepcopy(args)
                cfg = copy.deepcopy(default_config)
                cfg['jes_source'] = source
                cfg['jes'] = variation
                opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
                opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
                run(opts, configs={zjets_cfgname: cfg})

        # HEM15/16 unc
        if year == '2018':
            for variation in ['down']:
                syst_name = 'HEMIssue_%s' % variation
                logging.info('Start making %s trees...' % syst_name)
                opts = copy.deepcopy(args)
                cfg = copy.deepcopy(default_config)
                cfg['applyHEMUnc'] = True
                opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
                opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
                run(opts, configs={zjets_cfgname: cfg})


def _main(args):

    if not (args.post or args.add_weight or args.merge):
        tar_cmssw(args.tarball_suffix)

    if args.zjets_all:
        years = ['2015', '2016', '2017', '2018']
        categories = ['data', 'mc']
    else:
        years = args.year.split(',')
        categories = args.type.split(',')

    for year in years:
        for cat in categories:
            opts = copy.deepcopy(args)
            if cat == 'data':
                opts.nfiles_per_job *= 2
            if opts.inputdir:
                opts.inputdir = opts.inputdir.rstrip('/').replace('_YEAR_', year)
                print("INPUT DIR IS NOW:", opts.inputdir)
                assert(year in opts.inputdir)
                #base_dir_name = 'data' if cat == 'data' else 'mc'
                ## LG: this is a hack - fix it properly for the next round
                base_dir_name = 'data' if cat == 'data' else ''
                if opts.inputdir.rsplit('/', 1)[1] not in ['data', 'mc']:
                    opts.inputdir = os.path.join(opts.inputdir, base_dir_name)
                assert(opts.inputdir.endswith(base_dir_name))
            opts.type = cat
            opts.year = year
            logging.info('inputdir=%s, year=%s, cat=%s, syst=%s', opts.inputdir, opts.year,
                         opts.type, opts.zjets_do_syst)
            _process(opts)


def main():
    parser = get_arg_parser()

    #parser.add_argument(
    #    '--inputdir', required=True,
    #    help='Location of input samples')

    parser.add_argument(
        '--year', required=False,
        help='Year: 2015 (2016 preVFP), 2016 (2016 postVFP), 2017, 2018, or comma separated list e.g., `2016,2017,2018`')

    parser.add_argument(
        '--type', type=str,
        help='Run `data` or `mc` or `syst`, or a combination of them with a comma-separated string (e.g., `data,mc`)')

    parser.add_argument('--zjets-do-syst',
                        action='store_true', default=False,
                        help='Run all the systematic trees. Default: %(default)s'
                        )

    parser.add_argument('--zjets-all',
                        action='store_true', default=False,
                        help='Run over all three years and all channels. Default: %(default)s'
                        )

    parser.add_argument('--sample-dir',
                        type=str,
                        default='samples',
                        help='Directory of the sample list files. Default: %(default)s'
                        )

    parser.add_argument(
        '--wait', type=int, default=-1,
        help='To be used together with `--post --batch` to keep the postprocessing waiting until all jobs finished.'
        ' The value is the number of seconds to wait between two trials.')

    args = parser.parse_args()

    if args.wait > 0:
        if args.post or args.add_weight or args.merge:
            while True:
                _main(args)
                print('... waiting ...')
                time.sleep(args.wait)
    else:
        _main(args)


if __name__ == '__main__':
    main()
