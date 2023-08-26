import os
import correctionlib
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module


era_dict = {2015: '2016preVFP_UL', 2016: '2016postVFP_UL', 2017: '2017_UL', 2018: '2018_UL'}
tag_dict = {
    0: 'L0',
    40: 'C0', 41: 'C1', 42: 'C2', 43: 'C3', 44: 'C4',
    50: 'B0', 51: 'B1', 52: 'B2', 53: 'B3', 54: 'B4',
}
# FIXME: systematics list
systematics = [
    'Stat',
    'LHEScaleWeight_muF_ttbar',
    'LHEScaleWeight_muF_wjets',
    'LHEScaleWeight_muF_zjets',
    'LHEScaleWeight_muR_ttbar',
    'LHEScaleWeight_muR_wjets',
    'LHEScaleWeight_muR_zjets',
    'PSWeightISR',
    'PSWeightFSR',
    'XSec_WJets_c',
    'XSec_WJets_b',
    'XSec_ZJets_c',
    'XSec_ZJets_b',
    # 'JER',
    # 'JES',
    'PUWeight',
    'PUJetID'
]


# def debug(msg, activated=True):
#     if activated:
#         print(' > ', msg)


class FlavTagSFProducer(Module):

    def __init__(self, year):
        era = {2015: '2016preVFP_UL', 2016: '2016postVFP_UL', 2017: '2017_UL', 2018: '2018_UL'}[year]
        correction_file = os.path.expandvars(
            f'$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data/flavTagSF/flavTaggingSF_{era}.json.gz')
        self.corr = correctionlib.CorrectionSet.from_file(correction_file)['particleNetAK4_shape']

    def get_sf(self, j, syst='central'):
        return self.corr.evaluate(syst, j.hadronFlavour, tag_dict[j.tag], abs(j.eta), j.pt)

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        if self.isMC:
            self.out = wrappedOutputTree
            self.basewgts = {'flavTagWeight': 1.}
            for syst in systematics:
                self.basewgts[f'flavTagWeight_{syst}_UP'] = 1.
                self.basewgts[f'flavTagWeight_{syst}_DOWN'] = 1.
            for name in self.basewgts.keys():
                self.out.branch(name, "F")

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        if not self.isMC:
            return True

        jets = event.ak4jets
        wgts = self.basewgts.copy()

        # debug('------')
        for j in jets:
            # debug('jet pt:%.1f, eta:%.2f, flavor:%d, tag:%s, SF:%.4f' %
            #       (j.pt, j.eta, j.hadronFlavour, j.tag, self.get_sf(j)))
            wgts['flavTagWeight'] *= self.get_sf(j)
            for syst in systematics:
                wgts[f'flavTagWeight_{syst}_UP'] *= self.get_sf(j, 'up_' + syst)
                wgts[f'flavTagWeight_{syst}_DOWN'] *= self.get_sf(j, 'down_' + syst)
                # debug('syst:%s, up:%.4f, down:%.4f' %
                #       (syst, self.get_sf(j, 'up_' + syst), self.get_sf(j, 'down_' + syst)))

        # debug('Event wgt:\n - ' + ('\n - '.join(str(it) for it in wgts.items())))

        for name, val in wgts.items():
            self.out.fillBranch(name, val)

        return True


def flavTagSF_2015():
    return FlavTagSFProducer(year=2015)


def flavTagSF_2016():
    return FlavTagSFProducer(year=2016)


def flavTagSF_2017():
    return FlavTagSFProducer(year=2017)


def flavTagSF_2018():
    return FlavTagSFProducer(year=2018)
