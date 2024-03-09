import numpy as np
import os
import correctionlib
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module


era_dict = {2015: '2016preVFP_UL', 2016: '2016postVFP_UL', 2017: '2017_UL', 2018: '2018_UL'}


# def debug(msg, activated=False):
#     if activated:
#         print(' > ', msg)


class ElectronSFProducer(Module):

    def __init__(self, year, split_weights=False):
        self.year = year
        self.era = era_dict[self.year]
        correction_file = f'/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/{self.era}/electron.json.gz'
        self.corr = correctionlib.CorrectionSet.from_file(correction_file)['UL-Electron-ID-SF']
        self.split_weights = split_weights

    def get_sf(self, sf_type, lep):
        if abs(lep.pdgId) != 11:
            raise RuntimeError('Input lepton is not a electron')

        wp = None
        if sf_type == 'Reco':
            wp = 'RecoBelow20' if lep.pt < 20 else 'RecoAbove20'
        elif sf_type == 'ID':
            wp = lep._wp_ID

        scale_factors = np.array([self.corr.evaluate(self.era.replace('_UL', ''), syst, wp, lep.etaSC, lep.pt)
                                  for syst in ('sf', 'sfup', 'sfdown')])
        # debug(f'Electron pt:{lep.pt:.1f}, etaSC:{lep.etaSC:.2f}, {wp}, SF({sf_type}) = {scale_factors} (nom, up, down)')
        return scale_factors

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        if self.isMC:
            self.out = wrappedOutputTree

            if self.split_weights:
                self.out.branch('elRecoWeight', "F")
                self.out.branch('elRecoWeight_UP', "F")
                self.out.branch('elRecoWeight_DOWN', "F")

                self.out.branch('elIDWeight', "F")
                self.out.branch('elIDWeight_UP', "F")
                self.out.branch('elIDWeight_DOWN', "F")
            else:
                self.out.branch('elEffWeight', "F")
                self.out.branch('elEffWeight_UP', "F")
                self.out.branch('elEffWeight_DOWN', "F")

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        if not self.isMC:
            return True

        wgtReco = np.ones(3)
        wgtID = np.ones(3)

        for lep in event.selectedLeptons[:2]:
            # consider only up to two leading leptons
            if abs(lep.pdgId) != 11:
                continue
            wgtReco *= self.get_sf('Reco', lep)
            wgtID *= self.get_sf('ID', lep)

        if self.split_weights:
            self.out.fillBranch('elRecoWeight', wgtReco[0])
            self.out.fillBranch('elRecoWeight_UP', wgtReco[1])
            self.out.fillBranch('elRecoWeight_DOWN', wgtReco[2])

            self.out.fillBranch('elIDWeight', wgtID[0])
            self.out.fillBranch('elIDWeight_UP', wgtID[1])
            self.out.fillBranch('elIDWeight_DOWN', wgtID[2])
        else:
            eventWgt = wgtReco * wgtID
            self.out.fillBranch('elEffWeight', eventWgt[0])
            self.out.fillBranch('elEffWeight_UP', eventWgt[1])
            self.out.fillBranch('elEffWeight_DOWN', eventWgt[2])

        return True


class MuonSFProducer(Module):

    def __init__(self, year, split_weights=False):
        self.year = year
        self.era = era_dict[self.year]
        correction_file = f'/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/{self.era}/muon_Z.json.gz'
        self.corr = correctionlib.CorrectionSet.from_file(correction_file)
        self.split_weights = split_weights

    def get_sf(self, sf_type, lep):
        if abs(lep.pdgId) != 13:
            raise RuntimeError('Input lepton is not a muon')
        if sf_type == 'ID':
            assert lep._wp_ID == 'TightID'
            key = 'NUM_TightID_DEN_genTracks'
        elif sf_type == 'Iso':
            key = f'NUM_{lep._wp_Iso}_DEN_TightIDandIPCut'
        scale_factors = np.array([self.corr[key].evaluate(self.era, abs(lep.eta), lep.pt, syst)
                                  for syst in ('sf', 'systup', 'systdown')])
        # debug(f'Muon pt:{lep.pt:.1f}, eta:{lep.eta:.2f}, wp:({lep._wp_ID}, {lep._wp_Iso}), SF({sf_type}) = {scale_factors} (nom, up, down)')
        return scale_factors

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        if self.isMC:
            self.out = wrappedOutputTree

            if self.split_weights:
                self.out.branch('muIDWeight', "F")
                self.out.branch('muIDWeight_UP', "F")
                self.out.branch('muIDWeight_DOWN', "F")

                self.out.branch('muIsoWeight', "F")
                self.out.branch('muIsoWeight_UP', "F")
                self.out.branch('muIsoWeight_DOWN', "F")
            else:
                self.out.branch('muEffWeight', "F")
                self.out.branch('muEffWeight_UP', "F")
                self.out.branch('muEffWeight_DOWN', "F")

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        if not self.isMC:
            return True

        wgtID = np.ones(3)
        wgtIso = np.ones(3)

        for lep in event.selectedLeptons[:2]:
            # consider only up to two leading leptons
            if abs(lep.pdgId) != 13:
                continue
            wgtID *= self.get_sf('ID', lep)
            wgtIso *= self.get_sf('Iso', lep)

        if self.split_weights:
            self.out.fillBranch('muIDWeight', wgtID[0])
            self.out.fillBranch('muIDWeight_UP', wgtID[1])
            self.out.fillBranch('muIDWeight_DOWN', wgtID[2])

            self.out.fillBranch('muIsoWeight', wgtIso[0])
            self.out.fillBranch('muIsoWeight_UP', wgtIso[1])
            self.out.fillBranch('muIsoWeight_DOWN', wgtIso[2])
        else:
            eventWgt = wgtID * wgtIso
            self.out.fillBranch('muEffWeight', eventWgt[0])
            self.out.fillBranch('muEffWeight_UP', eventWgt[1])
            self.out.fillBranch('muEffWeight_DOWN', eventWgt[2])

        return True


class TriggerSF():

    def __init__(self, year, channel):
        self.era = era_dict[year]
        self.channel = channel
        if channel == '1L':
            self.corr_mu = correctionlib.CorrectionSet.from_file(
                f'/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/{self.era}/muon_Z.json.gz')[
                'NUM_%s_DEN_CutBasedIdTight_and_PFIsoTight' %
                ('IsoMu27' if year == 2017 else 'IsoMu24' if year == 2018 else 'IsoMu24_or_IsoTkMu24')]
            self.corr_el = correctionlib.CorrectionSet.from_file(os.path.expandvars(
                f'$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data/scale_factors/trigger/scale_factors_Ele_{self.era}.json'))
        elif channel == '2L':
            self.corr_ElEl = correctionlib.CorrectionSet.from_file(os.path.expandvars(
                f'$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data/scale_factors/trigger/scale_factors_ElEl_{self.era}.json'))
            self.corr_ElMu = correctionlib.CorrectionSet.from_file(os.path.expandvars(
                f'$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data/scale_factors/trigger/scale_factors_ElMu_{self.era}.json'))
            self.corr_MuMu = correctionlib.CorrectionSet.from_file(os.path.expandvars(
                f'$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data/scale_factors/trigger/scale_factors_MuMu_{self.era}.json'))

    def get_trigger_sf(self, event):
        trigWgt = np.ones(3)

        if self.channel == '1L':
            lep = event.selectedLeptons[0]
            if abs(lep.pdgId) == 13:
                trigWgt = np.array([self.corr_mu.evaluate(self.era, abs(lep.eta), lep.pt, syst)
                                    for syst in ('sf', 'systup', 'systdown')])
                # debug(f'Muon pt:{lep.pt:.1f}, eta:{lep.eta:.2f}, SF(trigger) = {trigWgt} (nom, up, down)')
            elif abs(lep.pdgId) == 11:
                trigWgt = np.array([self.corr_el[f'trigger_SF_{syst}'].evaluate(lep.pt, lep.etaSC)
                                    for syst in ('nom', 'up', 'down')])
                # debug(f'Electron pt:{lep.pt:.1f}, etaSC:{lep.etaSC:.2f}, SF(trigger) = {trigWgt} (nom, up, down)')

        elif self.channel == '2L':
            electrons = []
            muons = []
            for lep in event.selectedLeptons[:2]:
                if abs(lep.pdgId) == 11:
                    electrons.append(lep)
                elif abs(lep.pdgId) == 13:
                    muons.append(lep)

            if len(electrons) == 2:
                trigWgt = np.array([self.corr_ElEl[f'trigger_SF_{syst}'].evaluate(electrons[0].pt, electrons[1].pt)
                                    for syst in ('nom', 'up', 'down')])
            elif len(muons) == 2:
                trigWgt = np.array([self.corr_MuMu[f'trigger_SF_{syst}'].evaluate(muons[0].pt, muons[1].pt)
                                    for syst in ('nom', 'up', 'down')])
            elif len(electrons) == 1 and len(muons) == 1:
                trigWgt = np.array([self.corr_ElMu[f'trigger_SF_{syst}'].evaluate(muons[0].pt, electrons[0].pt)
                                    for syst in ('nom', 'up', 'down')])

            # debug(f'Leading (pdgId:{event.selectedLeptons[0].pdgId}, pt:{event.selectedLeptons[0].pt:.1f}, eta:{event.selectedLeptons[0].eta:.2f}), Sub-Leading (pdgId:{event.selectedLeptons[1].pdgId}, pt:{event.selectedLeptons[1].pt:.1f}, eta:{event.selectedLeptons[1].eta:.2f}), SF(trigger) = {trigWgt} (nom, up, down)')

        return trigWgt

# ====== electron ======


def electronSF_2015():
    return ElectronSFProducer(year=2015)


def electronSF_2016():
    return ElectronSFProducer(year=2016)


def electronSF_2017():
    return ElectronSFProducer(year=2017)


def electronSF_2018():
    return ElectronSFProducer(year=2018)


# ====== muon ======

def muonSF_2015():
    return MuonSFProducer(year=2015)


def muonSF_2016():
    return MuonSFProducer(year=2016)


def muonSF_2017():
    return MuonSFProducer(year=2017)


def muonSF_2018():
    return MuonSFProducer(year=2018)
