import ROOT
import os
import math
import numpy as np
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module


def debug(msg, activated=False):
    if activated:
        print(' > ', msg)


def get_sf_root(hist, x, y, inflate_unc_out_bounds=False, no_unc=False):
    bin_x0 = hist.GetXaxis().FindFixBin(x)
    bin_y0 = hist.GetYaxis().FindFixBin(y)
    binX = int(np.clip(bin_x0, 1, hist.GetNbinsX()))
    binY = int(np.clip(bin_y0, 1, hist.GetNbinsY()))
    val = hist.GetBinContent(binX, binY)
    if no_unc:
        return np.array([val, val, val])
    err = hist.GetBinError(binX, binY)
    if binX != bin_x0 or binY != bin_y0:
        if inflate_unc_out_bounds:
            err *= 2.
    return np.array([val, val + err, val - err])


class SFHelper(object):

    def __init__(self, prefix, file=None, key=None, systkey=None):
        if file is None:
            self.obj = None
            return

        filepath = os.path.join(prefix, file)
        if filepath.endswith('.root'):
            self._rtfile = ROOT.TFile(filepath)
            self.obj = self._rtfile.Get(key)
            self.syst = self._rtfile.Get(systkey) if systkey else None
        else:
            raise RuntimeError(
                'Invalid file %s. Only root files are supported.' % filepath)

    def __call__(self, x, y, inflate_unc_out_bounds=False, no_unc=False):
        if self.obj is None:
            return np.ones(3)
        else:
            if self.syst:
                val = get_sf_root(self.obj, x, y, inflate_unc_out_bounds=False, no_unc=True)[0]
                err = get_sf_root(self.syst, x, y, inflate_unc_out_bounds=False, no_unc=True)[0]
                if inflate_unc_out_bounds:
                    err *= 2
                return np.array([val, val + err, val - err])
            else:
                return get_sf_root(self.obj, x, y, inflate_unc_out_bounds, no_unc)



class PileupJetIdSFProducer(Module):

    def __init__(self, year, wp='T', wpsel=lambda j: j.puId == 7):
        self.year = int(year)
        basedir = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data/puJetId')
        self.cfg = {
            'effMC': {'file': 'effcyPUID_81Xtraining.root', 'key': 'h2_eff_mc{year}_{wp}'.format(year=year, wp=wp)},
            'mistagMC': {'file': 'effcyPUID_81Xtraining.root', 'key': 'h2_mistag_mc{year}_{wp}'.format(year=year, wp=wp)},
            'effSF': {'file': 'scalefactorsPUID_81Xtraining.root',
                      'key': 'h2_eff_sf{year}_{wp}'.format(year=year, wp=wp),
                      'systkey': 'h2_eff_sf{year}_{wp}_Systuncty'.format(year=year, wp=wp)},
            'mistagSF': {'file': 'scalefactorsPUID_81Xtraining.root',
                         'key': 'h2_mistag_sf{year}_{wp}'.format(year=year, wp=wp),
                         'systkey': 'h2_mistag_sf{year}_{wp}_Systuncty'.format(year=year, wp=wp)},
        }
        print('Init puJetId SF:', self.cfg)
        self.SFs = {k: SFHelper(prefix=basedir, **self.cfg[k]) for k in self.cfg}
        self.wpsel = wpsel

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        if self.isMC:
            self.out = wrappedOutputTree
            self.out.branch('pileupJetIdWeight', "F")
            self.out.branch('pileupJetIdWeight_UP', "F")
            self.out.branch('pileupJetIdWeight_DOWN', "F")


    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        if not self.isMC:
            return True

        allJets = Collection(event, "Jet")

        debug('------')
        prob_mc, prob_data = np.ones(3), np.ones(3)
        for j in allJets:
            if not (j.pt > 25 and j.pt < 50 and abs(j.eta) < 2.4 and (j.jetId & 4)):
                # pt, eta, tightIdLepVeto
                continue
            debug('jet pt:%.1f, eta:%.2f, genIdx:%d, pass puId:%s' %(j.pt, j.eta, j.genJetIdx, self.wpsel(j)))
            mc_obj, sf_obj = (self.SFs['effMC'], self.SFs['effSF']) if j.genJetIdx >= 0 else (self.SFs['mistagMC'], self.SFs['mistagSF'])
            eff = mc_obj(j.pt, j.eta, no_unc=True)
            sf = np.clip(sf_obj(j.pt, j.eta), 0, 2)
            debug('puId mc eff/mistag: %s, sf: %s' % (str(eff), str(sf)))
            if self.wpsel(j):
                prob_mc *= np.clip(eff, 1e-3, 1)
                prob_data *= np.clip(eff * sf, 1e-3, 1)
            else:
                prob_mc *= np.clip(1 - eff, 1e-3, 1)
                prob_data *= np.clip(1 - eff * sf, 1e-3, 1)
        eventWgt = np.clip(prob_data / prob_mc, 0, 2)
        debug('event prob_mc: %s, prob_data: %s, weight: %s' % (str(prob_mc), str(prob_data), str(eventWgt)))

        self.out.fillBranch('pileupJetIdWeight', eventWgt[0])
        self.out.fillBranch('pileupJetIdWeight_UP', eventWgt[1])
        self.out.fillBranch('pileupJetIdWeight_DOWN', eventWgt[2])

        return True


def puJetIdSF_2016():
    return PileupJetIdSFProducer(year=2016)


def puJetIdSF_2017():
    return PileupJetIdSFProducer(year=2017)


def puJetIdSF_2018():
    return PileupJetIdSFProducer(year=2018)
