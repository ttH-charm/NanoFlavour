import os
import correctionlib
import numpy as np
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module


# def debug(msg, activated=True):
#     if activated:
#         print(' > ', msg)


def rndSeed(event, jets, extra=0):
    seed = (event.run << 20) + (event.luminosityBlock << 10) + event.event + extra
    if len(jets) > 0:
        seed += int(jets[0].eta / 0.01)
    return seed


class FlavTagSFProducer(Module):

    def __init__(self, year, weight_name='flavTagWeight', split_stat_unc=True):
        era = {2015: '2016preVFP_UL', 2016: '2016postVFP_UL', 2017: '2017_UL', 2018: '2018_UL', 'Run2': 'Run2_UL'}[year]
        correction_file = os.path.expandvars(
            f'$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data/flavTagSF/flavTaggingSF_{era}.json.gz')
        self.corr = correctionlib.CorrectionSet.from_file(correction_file)['particleNetAK4_shape']
        self.name = weight_name

        self.systematics = [
            'Stat',
            'LHEScaleWeight_muF_ttbar',
            'LHEScaleWeight_muF_wjets',
            'LHEScaleWeight_muF_zjets',
            'LHEScaleWeight_muR_ttbar',
            'LHEScaleWeight_muR_wjets',
            'LHEScaleWeight_muR_zjets',
            'PSWeightISR_ttbar',
            'PSWeightISR_wjets',
            'PSWeightISR_zjets',
            'PSWeightFSR_ttbar',
            'PSWeightFSR_wjets',
            'PSWeightFSR_zjets',
            'XSec_WJets_c',
            'XSec_WJets_b',
            'XSec_ZJets_c',
            'XSec_ZJets_b',
            'JER',
            'JES',
            'PUWeight',
            # 'PUJetID'
        ]
        if split_stat_unc:
            flavors = ['flavB', 'flavC', 'flavL']
            tag_categories = ['C0', 'C1', 'C2', 'C3', 'C4', 'B0', 'B1', 'B2', 'B3', 'B4']
            for flav in flavors:
                for tag in tag_categories:
                    self.systematics.append(f'Stat_{flav}_{tag}')

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        if self.isMC:
            self.out = wrappedOutputTree
            self.basewgts = {self.name: 1.}
            for syst in self.systematics:
                self.basewgts[f'{self.name}_{syst}_UP'] = 1.
                self.basewgts[f'{self.name}_{syst}_DOWN'] = 1.
            for name in self.basewgts.keys():
                self.out.branch(name, "F")

            # # ====== testing ======
            # for name in self.basewgts.keys():
            #     self.out.branch('genMatched_' + name, "F")

    def compute_weights(self, event, jets, n_toys=1000):
        n = len(jets)
        hflav = np.zeros(n, dtype='int')
        wp = np.zeros(n, dtype='int')
        abseta = np.zeros(n, dtype='float')
        pt = np.zeros(n, dtype='float')
        for idx, j in enumerate(jets):
            hflav[idx] = j.hadronFlavour
            wp[idx] = j.tag
            abseta[idx] = abs(j.eta)
            pt[idx] = j.pt

        wgts = self.basewgts.copy()

        # central value
        sf_central = self.corr.evaluate('central', hflav, wp, abseta, pt)
        wgts[self.name] = sf_central.prod()

        # systematics
        for syst in self.systematics:
            if syst == 'Stat':
                # compute stat unc with toys
                sf_stat_up = self.corr.evaluate(f'up_{syst}', hflav, wp, abseta, pt)
                sf_stat_dn = self.corr.evaluate(f'down_{syst}', hflav, wp, abseta, pt)
                err = (np.abs(sf_stat_up - sf_central) + np.abs(sf_central - sf_stat_dn)) / 2
                # reset the seed to get reproducible results
                np.random.seed(rndSeed(event, jets))
                sf_toys = np.random.normal(sf_central[:, None], err[:, None], (n, n_toys))
                wgt_toys = np.clip(sf_toys, 0.3, 3).prod(axis=0)
                wgt_stat_dn, wgt_stat_up = np.percentile(wgt_toys, q=[16, 84])
                wgts[f'{self.name}_{syst}_UP'] = wgt_stat_up
                wgts[f'{self.name}_{syst}_DOWN'] = wgt_stat_dn
            else:
                wgts[f'{self.name}_{syst}_UP'] = self.corr.evaluate(f'up_{syst}', hflav, wp, abseta, pt).prod()
                wgts[f'{self.name}_{syst}_DOWN'] = self.corr.evaluate(f'down_{syst}', hflav, wp, abseta, pt).prod()

        # debug('------')
        # debug(f'- flav: {hflav}\n- wp: {wp}\n- abseta: {abseta}\n- pt: {pt}\n- sf: {sf_central}')
        # debug('Event weight:\n - ' + ('\n - '.join(str(it) for it in wgts.items())))

        return wgts

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        if not self.isMC:
            return True

        wgts = self.compute_weights(event, event.ak4jets)
        for name, val in wgts.items():
            self.out.fillBranch(name, np.clip(val, 0.3, 3))

        # # ====== testing ======
        # gen_matched_jets = [j for j in event.ak4jets if j.genJetIdx >= 0]
        # wgts_genMatched = self.compute_weights(event, gen_matched_jets)
        # for name, val in wgts_genMatched.items():
        #     self.out.fillBranch('genMatched_' + name, np.clip(val, 0.3, 3))

        return True


def flavTagSF_2015():
    return FlavTagSFProducer(year=2015)


def flavTagSF_2016():
    return FlavTagSFProducer(year=2016)


def flavTagSF_2017():
    return FlavTagSFProducer(year=2017)


def flavTagSF_2018():
    return FlavTagSFProducer(year=2018)


def flavTagSF_Run2():
    return FlavTagSFProducer(year='Run2')
