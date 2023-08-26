import numpy as np
import correctionlib
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module


'''https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL'''


# def debug(msg, activated=False):
#     if activated:
#         print(' > ', msg)


class PileupJetIdSFProducer(Module):

    def __init__(self, year, wp='T', wpsel=lambda j: j.puId >= 7):
        self.wp = wp
        self.wpsel = wpsel

        era = {2015: '2016preVFP_UL', 2016: '2016postVFP_UL', 2017: '2017_UL', 2018: '2018_UL'}[year]
        filename = f'/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/{era}/jmar.json.gz'
        self.corr = correctionlib.CorrectionSet.from_file(filename)['PUJetID_eff']

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

        # debug('------')
        # nom, up, down
        systs = ('nom', 'up', 'down')
        eventWgt = np.ones(3)
        for j in allJets:
            if not (j.pt > 25 and j.pt < 50 and abs(j.eta) < 2.4 and (j.jetId & 4)):
                # pt, eta, tightIdLepVeto
                continue
            # debug('jet pt:%.1f, eta:%.2f, genIdx:%d, pass puId:%s' % (j.pt, j.eta, j.genJetIdx, self.wpsel(j)))
            if not (j.genJetIdx >= 0 and self.wpsel(j)):
                # apply only to jets that passes the PileUpJetID and geometrically matched
                continue
            sf = np.array([self.corr.evaluate(j.eta, j.pt, syst, self.wp) for syst in systs])
            # debug('puId sf: %s' % str(sf))
            eventWgt *= sf
        eventWgt = np.clip(eventWgt, 0, 2)
        # debug('event weight: %s' % str(eventWgt))

        self.out.fillBranch('pileupJetIdWeight', eventWgt[0])
        self.out.fillBranch('pileupJetIdWeight_UP', eventWgt[1])
        self.out.fillBranch('pileupJetIdWeight_DOWN', eventWgt[2])

        return True


def puJetIdSF_2015():
    return PileupJetIdSFProducer(year=2015)


def puJetIdSF_2016():
    return PileupJetIdSFProducer(year=2016)


def puJetIdSF_2017():
    return PileupJetIdSFProducer(year=2017)


def puJetIdSF_2018():
    return PileupJetIdSFProducer(year=2018)
