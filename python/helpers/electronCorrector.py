from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
import os
import math
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from ..helpers.utils import polarP4


def rndSeed(event, obj):
    seed = (event.run << 20) + (event.luminosityBlock << 10) + event.event + int(obj.eta / 0.01)
    return seed


class ElectronScaleResCorrector(object):
    def __init__(self, year, corr='nominal'):
        self.year = year
        self.corr = corr

        if self.corr:
            fn = 'EgammaAnalysis/ElectronTools/data/ScalesSmearings/' + {
                2016: 'Legacy2016_07Aug2017_FineEtaR9_v3_ele_unc',
                2017: 'Run2017_17Nov2017_v1_ele_unc',
                2018: 'Run2018_Step2Closure_CoarseEtaR9Gain_v2',
            }[year]
            for library in ["libPhysicsToolsNanoFlavour"]:
                if library not in ROOT.gSystem.GetLibraries():
                    print("Load Library '%s'" % library.replace("lib", ""))
                    ROOT.gSystem.Load(library)
            print('Loading egamma scale and smearing file from %s' % fn)
            self.eleCorr = ROOT.EnergyScaleCorrection(fn, ROOT.EnergyScaleCorrection.ECALELF)
            self.rnd = ROOT.TRandom3()

    def correct(self, event, el, is_mc):
        if not is_mc:
            return
        if not self.corr or self.corr == 'nominal':
            return

        self.rnd.SetSeed(rndSeed(event, el))
        smearNrSigma = self.rnd.Gaus(0, 1)

        run = event.run
        abseta = abs(el.eta + el.deltaEtaSC)
        et = polarP4(el).energy() / math.cosh(abseta)
        r9 = el.r9
        seed = el.seedGain

        # print('et:', et, 'r9:', r9, 'abseta:', abseta, 'seed:', seed)

        smearCorr = self.eleCorr.getSmearCorr(run, et, abseta, r9, seed)
        if smearCorr:
            smear = smearCorr.sigma(et)
            smearUp = smearCorr.sigma(et, 1.0, 0.0)
            smearDn = smearCorr.sigma(et, -1.0, 0.0)
        else:
            smear = 0
            smearUp = 0
            smearDn = 0

        corrUpDelta = abs((smearUp - smear) * smearNrSigma)
        corrDnDelta = abs((smearDn - smear) * smearNrSigma)
        smearErr = max(corrUpDelta, corrDnDelta)
        scaleErr = self.eleCorr.scaleCorrUncert(run, et, abseta, r9, seed, ROOT.std.bitset(
            ROOT.EnergyScaleCorrection.kErrNrBits)(ROOT.EnergyScaleCorrection.kErrStatSystGain))
        err = math.hypot(smearErr, scaleErr)

        # print('smear:', smear, 'smearUp:', smearUp, 'smearDn:', smearDn, 'corr:',
        #       corr, 'smearErr:', smearErr, 'scaleErr:', scaleErr, 'err:', err)

        if self.corr == 'up':
            el.pt = el.pt / el.eCorr * (el.eCorr + err)
        elif self.corr == 'down':
            el.pt = el.pt / el.eCorr * (el.eCorr - err)

        # corr = 1.0 + smear * smearNrSigma
        # if self.corr == 'nominal':
        #     el.pt = el.pt / el.eCorr * corr
        # elif self.corr == 'up':
        #     el.pt = el.pt / el.eCorr * (corr + err)
        # elif self.corr == 'down':
        #     el.pt = el.pt / el.eCorr * (corr - err)
