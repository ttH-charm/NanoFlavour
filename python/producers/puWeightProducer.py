import numpy as np
import os
import correctionlib
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module


era_dict = {2015: '2016preVFP_UL', 2016: '2016postVFP_UL', 2017: '2017_UL', 2018: '2018_UL'}
key_dict = {2015: 'Collisions16_UltraLegacy_goldenJSON',
            2016: 'Collisions16_UltraLegacy_goldenJSON',
            2017: 'Collisions17_UltraLegacy_goldenJSON',
            2018: 'Collisions18_UltraLegacy_goldenJSON'}


class PileupWeightProducer(Module):
    def __init__(self, year, nvtxVar='Pileup_nTrueInt'):
        self.year = year
        self.nvtxVar = nvtxVar
        self.era = era_dict[self.year]
        correction_file = f'/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/LUM/{self.era}/puWeights.json.gz'
        self.corr = correctionlib.CorrectionSet.from_file(correction_file)[key_dict[self.year]]

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        if self.isMC:
            self.out = wrappedOutputTree

            self.out.branch('puWeight', "F")
            self.out.branch('puWeightUp', "F")
            self.out.branch('puWeightDown', "F")

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        if not self.isMC:
            return True

        puWeights = [self.corr.evaluate(getattr(event, self.nvtxVar), syst)
                     for syst in ('nominal', 'up', 'down')]

        self.out.fillBranch('puWeight', puWeights[0])
        self.out.fillBranch('puWeightUp', puWeights[1])
        self.out.fillBranch('puWeightDown', puWeights[2])
        return True


def puWeight_2015():
    return PileupWeightProducer(year=2015)


def puWeight_2016():
    return PileupWeightProducer(year=2016)


def puWeight_2017():
    return PileupWeightProducer(year=2017)


def puWeight_2018():
    return PileupWeightProducer(year=2018)
