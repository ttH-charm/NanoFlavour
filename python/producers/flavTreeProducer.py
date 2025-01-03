import os
import re
import numpy as np
import math
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from ..helpers.utils import deltaPhi, deltaR, deltaR2, deltaEta, closest, polarP4, sumP4, transverseMass, minValue, configLogger, getDigit, closest_pair
from ..helpers.nnHelper import convert_prob
from ..helpers.jetmetCorrector import JetMETCorrector, rndSeed
from ..helpers.muonCorrector import MuonScaleResCorrector
from ..helpers.triggerHelper import passTrigger
from .leptonSFProducer import TriggerSF

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

lumi_dict = {2015: 19.52, 2016: 16.81, 2017: 41.48, 2018: 59.83}
channel_dict = {'ZJets': 0, 'WJets': 1, 'TT1L': 2, 'TT2L': 3}
dataset_dict = {
    'SingleMuon': 100001,
    'SingleElectron': 100002,
    'MuonEG': 100003,
    'DoubleMuon': 100004,
    'DoubleEG': 100005,
    'EGamma': 100006,
    'JetHT': 100007,
    'MET': 100008,
    'BtagCSV': 100009,
}


class METObject(Object):

    def p4(self):
        return polarP4(self, eta=None, mass=None)


class FlavTreeProducer(Module, object):

    def __init__(self, channel, **kwargs):
        self._channel = channel
        self._chn_code = channel_dict[channel]
        self._year = int(kwargs['year'])
        self._usePuppiJets = kwargs['usePuppiJets']
        self._jmeSysts = {'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': '',
                          'jer': 'nominal', 'jmr': None, 'met_unclustered': None, 'applyHEMUnc': False,
                          'smearMET': False}
        self._opts = {'muon_scale': 'nominal', 'fillJetTaggingScores': False,
                      'apply_tight_selection': True, 'for_training': False}
        for k in kwargs:
            if k in self._jmeSysts:
                self._jmeSysts[k] = kwargs[k]
            else:
                self._opts[k] = kwargs[k]
        self._needsJMECorr = any([self._jmeSysts['jec'], self._jmeSysts['jes'],
                                  self._jmeSysts['jer'], self._jmeSysts['jmr'],
                                  self._jmeSysts['met_unclustered'], self._jmeSysts['applyHEMUnc']])

        logger.info('Running %s channel for year %s with JME systematics %s, other options %s',
                    self._channel, str(self._year), str(self._jmeSysts), str(self._opts))

        if self._needsJMECorr:
            self.jetmetCorr = JetMETCorrector(
                year=self._year, jetType="AK4PFPuppi" if self._usePuppiJets else "AK4PFchs", **self._jmeSysts)

        self.muonCorr = MuonScaleResCorrector(year=self._year, corr=self._opts['muon_scale'])

        # ParticleNetAK4 -- exclusive b- and c-tagging categories
        # 5x: b-tagged; 4x: c-tagged; 0: light
        if self._year in (2017, 2018):
            self.jetTagWPs = {
                54: '(pn_b_plus_c>0.5) & (pn_b_vs_c>0.99)',
                53: '(pn_b_plus_c>0.5) & (0.96<pn_b_vs_c<=0.99)',
                52: '(pn_b_plus_c>0.5) & (0.88<pn_b_vs_c<=0.96)',
                51: '(pn_b_plus_c>0.5) & (0.70<pn_b_vs_c<=0.88)',
                50: '(pn_b_plus_c>0.5) & (0.40<pn_b_vs_c<=0.70)',

                44: '(pn_b_plus_c>0.5) & (pn_b_vs_c<=0.05)',
                43: '(pn_b_plus_c>0.5) & (0.05<pn_b_vs_c<=0.15)',
                42: '(pn_b_plus_c>0.5) & (0.15<pn_b_vs_c<=0.40)',
                41: '(0.2<pn_b_plus_c<=0.5)',
                40: '(0.1<pn_b_plus_c<=0.2)',

                0: '(pn_b_plus_c<=0.1)',
            }
        elif self._year in (2015, 2016):
            self.jetTagWPs = {
                54: '(pn_b_plus_c>0.35) & (pn_b_vs_c>0.99)',
                53: '(pn_b_plus_c>0.35) & (0.96<pn_b_vs_c<=0.99)',
                52: '(pn_b_plus_c>0.35) & (0.88<pn_b_vs_c<=0.96)',
                51: '(pn_b_plus_c>0.35) & (0.70<pn_b_vs_c<=0.88)',
                50: '(pn_b_plus_c>0.35) & (0.40<pn_b_vs_c<=0.70)',

                44: '(pn_b_plus_c>0.35) & (pn_b_vs_c<=0.05)',
                43: '(pn_b_plus_c>0.35) & (0.05<pn_b_vs_c<=0.15)',
                42: '(pn_b_plus_c>0.35) & (0.15<pn_b_vs_c<=0.40)',
                41: '(0.17<pn_b_plus_c<=0.35)',
                40: '(0.1<pn_b_plus_c<=0.17)',

                0: '(pn_b_plus_c<=0.1)',
            }

        if self._usePuppiJets:
            self.puID_WP = None
        else:
            # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL
            # self.puID_WP = {2015: -1, 2016: -1, 2017: -1, 2018: -1}[self._year]  # None
            # self.puID_WP = {2015: 1, 2016: 1, 2017: 4, 2018: 4}[self._year]  # L
            # self.puID_WP = {2015: 3, 2016: 3, 2017: 6, 2018: 6}[self._year]  # M
            self.puID_WP = {2015: 7, 2016: 7, 2017: 7, 2018: 7}[self._year]  # T

        self._trigSF = TriggerSF(self._year, '1L' if self._channel in ('WJets', 'TT1L') else '2L')

        logger.info('Running %s channel for year %s with jet tagging WPs %s, jet PU ID WPs %s',
                    self._channel, str(self._year), str(self.jetTagWPs), str(self.puID_WP))

    def evalJetTag(self, j, default=0):
        for wp, expr in self.jetTagWPs.items():
            if eval(expr, j.__dict__):
                return wp
        return default

    def beginJob(self):
        if self._needsJMECorr:
            self.jetmetCorr.beginJob()

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        self.hasParticleNetAK4 = 'privateNano' if inputTree.GetBranch(
            'Jet_ParticleNetAK4_probb') else 'jmeNano' if inputTree.GetBranch('Jet_particleNetAK4_B') else None
        if not self.hasParticleNetAK4:
            raise RuntimeError('No ParticleNetAK4 scores in the input NanoAOD!')
        self.rho_branch_name = 'Rho_fixedGridRhoFastjetAll' if bool(
            inputTree.GetBranch('Rho_fixedGridRhoFastjetAll')) else 'fixedGridRhoFastjetAll'

        self.dataset = None
        r = re.search(('mc' if self.isMC else 'data') + r'\/([a-zA-Z0-9_\-]+)\/', inputFile.GetName())
        if r:
            self.dataset = r.groups()[0]

        self.out = wrappedOutputTree

        # NOTE: branch names must start with a lower case letter
        # check keep_and_drop_output.txt
        self.out.branch("dataset", "I", title=', '.join([f'{k}={v}' for k, v in dataset_dict.items()]))
        self.out.branch("year", "I")
        self.out.branch("channel", "I")
        self.out.branch("lumiwgt", "F")

        self.out.branch("passmetfilters", "O")

        # triggers for 1L
        self.out.branch("passTrigEl", "O")
        self.out.branch("passTrigMu", "O")

        # triggers for 2L
        self.out.branch("passTrigElEl", "O")
        self.out.branch("passTrigElMu", "O")
        self.out.branch("passTrigMuMu", "O")
        # extra single e/mu trigger for 2L
        self.out.branch("passTrig2L_extEl", "O")
        self.out.branch("passTrig2L_extMu", "O")

        if self.isMC:
            self.out.branch("trigEffWeight", "F")
            self.out.branch("trigEffWeightUp", "F")
            self.out.branch("trigEffWeightDown", "F")

        self.out.branch("l1PreFiringWeight", "F")
        self.out.branch("l1PreFiringWeightUp", "F")
        self.out.branch("l1PreFiringWeightDown", "F")

        self.out.branch("met", "F")
        self.out.branch("met_phi", "F")
        self.out.branch("dphi_met_tkmet", "F")
        self.out.branch("min_dphi_met_jet", "F")

        # V boson
        self.out.branch("v_pt", "F")
        self.out.branch("v_eta", "F")
        self.out.branch("v_phi", "F")
        self.out.branch("v_mass", "F")

        # leptons
        self.out.branch("lep1_pt", "F")
        self.out.branch("lep1_eta", "F")
        self.out.branch("lep1_phi", "F")
        self.out.branch("lep1_mass", "F")
        self.out.branch("lep1_pfIso", "F")
        self.out.branch("lep1_miniIso", "F")
        self.out.branch("lep1_pdgId", "I")

        # self.out.branch("lep1_dxy", "F")
        # self.out.branch("lep1_dxyErr", "F")
        # self.out.branch("lep1_dz", "F")
        # self.out.branch("lep1_dzErr", "F")
        # self.out.branch("lep1_ip3d", "F")
        # self.out.branch("lep1_sip3d", "F")

        if self._channel in ('ZJets', 'TT2L'):
            self.out.branch("lep2_pt", "F")
            self.out.branch("lep2_eta", "F")
            self.out.branch("lep2_phi", "F")
            self.out.branch("lep2_mass", "F")
            self.out.branch("lep2_pdgId", "I")

        # event level
        if self._channel in ('WJets', 'TT1L'):
            self.out.branch("dphi_lep_met", "F")
            self.out.branch("mt_lep_met", "F")
        elif self._channel in ('ZJets', 'TT2L'):
            self.out.branch("mt_ll_met", "F")

        # ak4 jets
        self.out.branch("n_btag", "I")
        self.out.branch("n_ctag", "I")
        self.out.branch("n_mutag", "I")
        self.out.branch("ak4_pt", "F", 20, lenVar="n_ak4")
        self.out.branch("ak4_eta", "F", 20, lenVar="n_ak4")
        self.out.branch("ak4_phi", "F", 20, lenVar="n_ak4")
        self.out.branch("ak4_mass", "F", 20, lenVar="n_ak4")
        self.out.branch("ak4_tag", "I", 20, lenVar="n_ak4")
        self.out.branch("ak4_mu_ptfrac", "F", 20, lenVar="n_ak4")
        self.out.branch("ak4_mu_plus_nem", "F", 20, lenVar="n_ak4")
        self.out.branch("ak4_mu_pdgId", "I", 20, lenVar="n_ak4")
        if self.isMC:
            self.out.branch("ak4_hflav", "I", 20, lenVar="n_ak4")
            self.out.branch("ak4_pflav", "I", 20, lenVar="n_ak4")
            self.out.branch("ak4_genmatch", "I", 20, lenVar="n_ak4")
            self.out.branch("ak4_nBHadrons", "I", 20, lenVar="n_ak4")
            self.out.branch("ak4_nCHadrons", "I", 20, lenVar="n_ak4")

        if self._opts['fillJetTaggingScores']:
            # self.out.branch("ak4_bdisc", "F", 20, lenVar="n_ak4")
            # self.out.branch("ak4_cvbdisc", "F", 20, lenVar="n_ak4")
            # self.out.branch("ak4_cvldisc", "F", 20, lenVar="n_ak4")
            if self.hasParticleNetAK4:
                self.out.branch("ak4_prob_b", "F", 20, lenVar="n_ak4")
                self.out.branch("ak4_prob_bb", "F", 20, lenVar="n_ak4")
                self.out.branch("ak4_prob_c", "F", 20, lenVar="n_ak4")
                self.out.branch("ak4_prob_cc", "F", 20, lenVar="n_ak4")
                self.out.branch("ak4_prob_uds", "F", 20, lenVar="n_ak4")
                self.out.branch("ak4_prob_g", "F", 20, lenVar="n_ak4")
                self.out.branch("ak4_prob_pu", "F", 20, lenVar="n_ak4")
                self.out.branch("ak4_prob_undef", "F", 20, lenVar="n_ak4")

        self.out.branch("ht", "F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def _correctJetAndMET(self, event):
        if self._needsJMECorr:
            rho = getattr(event, self.rho_branch_name)
            # correct AK4 jets and MET
            self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
            self.jetmetCorr.correctJetAndMET(
                jets=event._allJets,
                lowPtJets=Collection(event, "CorrT1METJet"),
                met=event.met,
                rawMET=METObject(event, "RawPuppiMET") if self._usePuppiJets else METObject(event, "RawMET"),
                defaultMET=METObject(event, "PuppiMET") if self._usePuppiJets else METObject(event, "MET"),
                rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                isMC=self.isMC, runNumber=event.run)
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)  # sort by pt after updating

    def _selectLeptons(self, event):
        # do lepton selection
        event.looseLeptons = []  # used for jet lepton cleaning & lepton counting
        event.soft_muon_dict = {}

        electrons = Collection(event, "Electron")
        for el in electrons:
            el.etaSC = el.eta + el.deltaEtaSC
            # ttH(bb) analysis uses tight electron ID
            # if el.pt > 15 and abs(el.eta) < 2.4 and el.cutBased == 4:
            # NOTE: try mvaFall17V2Iso_WP90
            if 1.4442 <= abs(el.etaSC) <= 1.5560:
                continue
            if el.pt > 15 and abs(el.eta) < 2.4 and el.mvaIso_WP90:
                el._wp_ID = 'wp90iso'
                event.looseLeptons.append(el)

        muons = Collection(event, "Muon")
        for idx, mu in enumerate(muons):
            self.muonCorr.correct(event, mu, self.isMC)
            if mu.pt > 15 and abs(mu.eta) < 2.4 and mu.tightId and mu.pfRelIso04_all < 0.25:
                mu._wp_ID = 'TightID'
                mu._wp_Iso = 'LooseRelIso'
                event.looseLeptons.append(mu)
            elif mu.pt > 5 and abs(mu.eta) < 2.4 and mu.tightId and mu.pfRelIso04_all > 0.25:
                event.soft_muon_dict[idx] = mu

        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)

    def _preSelect(self, event):
        event.selectedLeptons = []  # used for reconstructing the W/Z boson / top quarks
        if self._channel in ('WJets', 'TT1L'):
            if len(event.looseLeptons) != 1:
                return False
            lep = event.looseLeptons[0]
            if abs(lep.pdgId) == 13:
                # mu (26/29/26 GeV)
                muPtCut = 29 if self._year == 2017 else 26
                if lep.pt > muPtCut and lep.tightId and lep.pfRelIso04_all < 0.15:
                    lep._wp_Iso = 'TightRelIso'
                    event.selectedLeptons.append(lep)
            else:
                # ele (29/30/30 GeV)
                ePtCut = 30 if self._year in (2017, 2018) else 29
                # ttH(bb) analysis uses tight electron ID
                # if lep.pt > ePtCut and lep.cutBased == 4:
                # NOTE: try mvaFall17V2Iso_WP80
                if lep.pt > ePtCut and lep.mvaIso_WP80:
                    lep._wp_ID = 'wp80iso'
                    event.selectedLeptons.append(lep)
            if len(event.selectedLeptons) != 1:
                return False
            # reject DY(-> mu mu)
            if abs(lep.pdgId) == 13:
                for soft_mu in event.soft_muon_dict.values():
                    mll = sumP4(lep, soft_mu).mass()
                    if mll < 12 or (81 < mll < 101):
                        return False
            if self._channel == 'WJets':
                if self._opts['apply_tight_selection']:
                    if len(event.soft_muon_dict) == 0:
                        return False
        elif self._channel in ('ZJets', 'TT2L'):
            if len(event.looseLeptons) != 2:
                return False
            for lep in event.looseLeptons:
                event.selectedLeptons.append(lep)
            if len(event.selectedLeptons) != 2:
                return False
            if event.selectedLeptons[0].pt < 25:
                return False
            if event.selectedLeptons[0].pdgId * event.selectedLeptons[1].pdgId > 0:
                # keep only opposite-sign
                return False
            if self._channel == 'TT2L':
                # ele+mu
                if event.selectedLeptons[0].pdgId + event.selectedLeptons[1].pdgId == 0:
                    return False
            elif self._channel == 'ZJets':
                if event.selectedLeptons[0].pdgId + event.selectedLeptons[1].pdgId != 0:
                    # (opposite-sign) same-flavor
                    return False
                Vboson = sumP4(event.selectedLeptons[0], event.selectedLeptons[1])
                if Vboson.mass() < 81 or Vboson.mass() > 101:
                    return False

        return True

    def _cleanObjects(self, event):
        event.ak4jets = []
        for j in event._allJets:
            if not (j.pt > 25 and abs(j.eta) < 2.4 and (j.jetId & 4)):
                # NOTE: ttH(bb) uses jets w/ pT > 30 GeV, loose PU Id
                # pt, eta, tightIdLepVeto, loose PU ID
                continue
            if not self._usePuppiJets and not (j.pt > 50 or j.puId >= self.puID_WP):
                # apply jet puId only for CHS jets
                continue
            if closest(j, event.looseLeptons)[1] < 0.4:
                continue
            j.btagDeepFlavC = j.btagDeepFlavB * j.btagDeepFlavCvB / (
                1 - j.btagDeepFlavCvB) if (j.btagDeepFlavCvB >= 0 and j.btagDeepFlavCvB < 1) else -1
            if self.hasParticleNetAK4 == 'privateNano':
                # attach ParticleNet scores
                j.pn_b = convert_prob(j, ['b', 'bb'], ['c', 'cc', 'uds', 'g'], 'ParticleNetAK4_prob')
                j.pn_c = convert_prob(j, ['c', 'cc'], ['b', 'bb', 'uds', 'g'], 'ParticleNetAK4_prob')
                j.pn_uds = convert_prob(j, 'uds', ['b', 'bb', 'c', 'cc', 'g'], 'ParticleNetAK4_prob')
                j.pn_g = convert_prob(j, 'g', ['b', 'bb', 'c', 'cc', 'uds'], 'ParticleNetAK4_prob')
                j.pn_b_plus_c = j.pn_b + j.pn_c
                j.pn_b_vs_c = j.pn_b / j.pn_b_plus_c
                j.tag = self.evalJetTag(j)
            elif self.hasParticleNetAK4 == 'jmeNano':
                # attach ParticleNet scores
                j.pn_b = j.particleNetAK4_B
                j.pn_c = j.particleNetAK4_B * j.particleNetAK4_CvsB / (
                    1 - j.particleNetAK4_CvsB) if (j.particleNetAK4_CvsB >= 0 and j.particleNetAK4_CvsB < 1) else -1
                j.pn_uds = np.clip(1 - j.pn_b - j.pn_c, 0, 1) * j.particleNetAK4_QvsG if (
                    j.particleNetAK4_QvsG >= 0 and j.particleNetAK4_QvsG < 1) else -1
                j.pn_g = np.clip(1 - j.pn_b - j.pn_c - j.pn_uds, 0, 1) if (
                    j.particleNetAK4_QvsG >= 0 and j.particleNetAK4_QvsG < 1) else -1
                j.pn_b_plus_c = j.pn_b + j.pn_c
                j.pn_b_vs_c = j.pn_b / j.pn_b_plus_c
                j.tag = self.evalJetTag(j)
            else:
                j.tag = 0

            # attach soft muon to jet
            muons_in_jet = [event.soft_muon_dict[i] for i in (j.muonIdx1, j.muonIdx2) if i in event.soft_muon_dict]
            if len(muons_in_jet):
                j.mu = max(muons_in_jet, key=lambda x: x.pt)
            else:
                j.mu = None
            event.ak4jets.append(j)

        event.ak4_b_jets = []
        event.ak4_c_jets = []

        for jet_idx, j in enumerate(event.ak4jets):
            j.idx = jet_idx
            if j.tag > 0:
                if j.tag >= 50:
                    event.ak4_b_jets.append(j)
                if 40 <= j.tag < 50:
                    event.ak4_c_jets.append(j)

    def _selectEvent(self, event):
        # logger.debug('processing event %d' % event.event)
        event.Vboson = None
        if self._channel in ('WJets', 'TT1L'):
            event.Vboson = polarP4(event.selectedLeptons[0]) + (event.met.p4())
        elif self._channel in ('ZJets', 'TT2L'):
            event.Vboson = sumP4(event.selectedLeptons[0], event.selectedLeptons[1])

        # channel specific selections
        if self._channel == 'ZJets':
            # 1 <= njets <= 2
            if not (1 <= len(event.ak4jets) <= 2):
                return False
            if self._opts['apply_tight_selection']:
                # n_ak4==1 && ak4_pt[0]/v_pt>0.5 && ak4_pt[0]/v_pt<2 && absDeltaPhi(v_phi, ak4_phi[0])>2
                if len(event.ak4jets) != 1:
                    return False
                if event.Vboson.pt() < 25:
                    return False
                if not (0.75 < event.ak4jets[0].pt / event.Vboson.pt() < 1.25):
                    return False
                if abs(deltaPhi(event.ak4jets[0].phi, event.Vboson.phi())) < 2:
                    return False
        elif self._channel == 'WJets':
            # # 1 <= njets <= 2
            # if not (1 <= len(event.ak4jets) <= 2):
            # njets == 1
            if len(event.ak4jets) != 1:
                return False
            if self._opts['apply_tight_selection']:
                # n_ak4==1 && met>30 && mt_lep_met>40 && mt_lep_met<120 && v_pt>30
                # && dphi_met_tkmet<1 && min_dphi_met_jet>1 "
                # && absDeltaPhi(v_phi, ak4_phi[0])>2 && ak4_pt[0]/v_pt>0.5 && ak4_pt[0]/v_pt<2
                if event.met.pt < 30:
                    return False
                if event.Vboson.pt() < 30:
                    return False
                if not (40 < transverseMass(event.selectedLeptons[0], event.met) < 120):
                    return False
                if abs(deltaPhi(event.met.phi, event.TkMET_phi)) > 1:
                    return False
                if abs(deltaPhi(event.met, event.ak4jets[0])) < 1:
                    return False
                if not (0.5 < event.ak4jets[0].pt / event.Vboson.pt() < 2):
                    return False
                if abs(deltaPhi(event.ak4jets[0].phi, event.Vboson.phi())) < 2:
                    return False
        elif self._channel == 'TT1L':
            # 3 <= njets <= 4
            if not (3 <= len(event.ak4jets) <= 4):
                return False
            if not self._opts['for_training']:
                if not (event.ak4jets[0].tag >= 50 or event.ak4jets[1].tag >= 50):
                    return False
            if self._opts['apply_tight_selection']:
                if event.met.pt < 20:
                    return False
        elif self._channel == 'TT2L':
            # 1 <= njets <= 2 ?? or ==2?
            if not (1 <= len(event.ak4jets) <= 2):
                return False
            if self._opts['apply_tight_selection']:
                # mt_ll_met>90 && n_ak4==2
                if len(event.ak4jets) != 2:
                    return False
                if event.met.pt < 20:
                    return False
                if transverseMass(event.Vboson, event.met) < 100:
                    return False

        # return True if passes selection
        return True

    def _selectTriggers(self, event):
        out_data = {}

        # !!! NOTE: make sure to update `keep_and_drop_input.txt` !!!
        if self._year <= 2016:
            out_data["passTrigEl"] = passTrigger(event, 'HLT_Ele27_WPTight_Gsf')
            out_data["passTrigMu"] = passTrigger(event, ['HLT_IsoMu24', 'HLT_IsoTkMu24'])
            out_data["passTrigElEl"] = passTrigger(event, 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ')
            out_data["passTrigElMu"] = passTrigger(event,
                                                   ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                                                    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL',
                                                    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ'])
            out_data["passTrigMuMu"] = (
                passTrigger(event, ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL'])
                if event.run <= 280385 else  # Run2016G
                passTrigger(event, ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ'])
            )
            out_data["passTrig2L_extEl"] = passTrigger(event, 'HLT_Ele27_WPTight_Gsf')
            out_data["passTrig2L_extMu"] = passTrigger(event, ['HLT_IsoMu24', 'HLT_IsoTkMu24'])
        elif self._year == 2017:
            flagL1DoubleEG = False
            for obj in Collection(event, "TrigObj"):
                if (obj.id == 11) and (obj.filterBits & 1024):
                    # 1024 = 1e (32_L1DoubleEG_AND_L1SingleEGOr)
                    flagL1DoubleEG = True
                    break
            event.HLT_Ele32_WPTight_Gsf_L1DoubleEG_L1Flag = event.HLT_Ele32_WPTight_Gsf_L1DoubleEG and flagL1DoubleEG
            out_data["passTrigEl"] = passTrigger(
                event, ['HLT_Ele32_WPTight_Gsf_L1DoubleEG_L1Flag', 'HLT_Ele28_eta2p1_WPTight_Gsf_HT150'])
            out_data["passTrigMu"] = passTrigger(event, 'HLT_IsoMu27')
            out_data["passTrigElEl"] = passTrigger(event, ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                           'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'])
            out_data["passTrigElMu"] = passTrigger(event,
                                                   ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                                                    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
                                                    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ'])
            out_data["passTrigMuMu"] = (
                passTrigger(event, 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ')
                if event.run <= 299329 else  # Run2017B
                passTrigger(event, 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8'))
            out_data["passTrig2L_extEl"] = passTrigger(event, 'HLT_Ele32_WPTight_Gsf_L1DoubleEG_L1Flag')
            out_data["passTrig2L_extMu"] = passTrigger(event, ['HLT_IsoMu24_eta2p1', 'HLT_IsoMu27'])
        elif self._year == 2018:
            out_data["passTrigEl"] = passTrigger(event,
                                                 ['HLT_Ele32_WPTight_Gsf',
                                                  'HLT_Ele28_eta2p1_WPTight_Gsf_HT150'])
            out_data["passTrigMu"] = passTrigger(event, 'HLT_IsoMu24')
            out_data["passTrigElEl"] = passTrigger(event,
                                                   ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'])
            out_data["passTrigElMu"] = passTrigger(event,
                                                   ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                                                    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
                                                    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ'])
            out_data["passTrigMuMu"] = passTrigger(event,
                                                   ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8',
                                                    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8'])
            out_data["passTrig2L_extEl"] = passTrigger(event, 'HLT_Ele32_WPTight_Gsf')
            out_data["passTrig2L_extMu"] = passTrigger(event, 'HLT_IsoMu24')

        # apply trigger selections on data
        if not self.isMC and self.dataset is not None:
            if self._channel in ('WJets', 'TT1L'):
                passTrig1L = False
                if self.dataset in ('EGamma', 'SingleElectron'):
                    passTrig1L = out_data['passTrigEl']
                elif self.dataset == 'SingleMuon':
                    passTrig1L = out_data['passTrigMu']
                if not passTrig1L:
                    return False

            elif self._channel in ('ZJets', 'TT2L'):
                passTrig2L = False
                if abs(event.selectedLeptons[0].pdgId) == 11 and abs(event.selectedLeptons[1].pdgId) == 11:
                    # ee channel
                    if self._year == 2018:
                        if self.dataset == 'EGamma':
                            passTrig2L = out_data["passTrigElEl"] or out_data['passTrig2L_extEl']
                    else:
                        if self.dataset == 'DoubleEG':
                            passTrig2L = out_data["passTrigElEl"]
                        elif self.dataset == 'SingleElectron':
                            passTrig2L = (not out_data["passTrigElEl"]) and out_data["passTrig2L_extEl"]
                elif abs(event.selectedLeptons[0].pdgId) == 13 and abs(event.selectedLeptons[1].pdgId) == 13:
                    # mumu channel
                    if self.dataset == 'DoubleMuon':
                        passTrig2L = out_data["passTrigMuMu"]
                    elif self.dataset == 'SingleMuon':
                        passTrig2L = (not out_data["passTrigMuMu"]) and out_data["passTrig2L_extMu"]
                else:
                    # emu channel
                    if self.dataset == 'MuonEG':
                        passTrig2L = out_data["passTrigElMu"]
                    elif self.dataset == 'SingleMuon':
                        passTrig2L = (not out_data["passTrigElMu"]) and out_data["passTrig2L_extMu"]
                    elif self.dataset in ('EGamma', 'SingleElectron'):
                        passTrig2L = (not out_data["passTrigElMu"]) and (
                            not out_data["passTrig2L_extMu"]) and out_data["passTrig2L_extEl"]

                if not passTrig2L:
                    return False

        for key in out_data:
            self.out.fillBranch(key, out_data[key])

        return True

    def _fillEventInfo(self, event):
        self.out.fillBranch("dataset", dataset_dict[self.dataset]
                            if self.dataset in dataset_dict else 1 if self.dataset else 0)
        self.out.fillBranch("year", self._year)
        self.out.fillBranch("channel", self._chn_code)
        self.out.fillBranch("lumiwgt", lumi_dict[self._year])

        # met filters -- updated for UL
        met_filters = bool(
            event.Flag_goodVertices and
            event.Flag_globalSuperTightHalo2016Filter and
            event.Flag_HBHENoiseFilter and
            event.Flag_HBHENoiseIsoFilter and
            event.Flag_EcalDeadCellTriggerPrimitiveFilter and
            event.Flag_BadPFMuonFilter and
            event.Flag_BadPFMuonDzFilter and
            event.Flag_eeBadScFilter
        )
        if self._year in (2017, 2018):
            met_filters = met_filters and event.Flag_ecalBadCalibFilter
        self.out.fillBranch("passmetfilters", met_filters)

        # trigger SFs
        if self.isMC:
            trigWgt = self._trigSF.get_trigger_sf(event)
            self.out.fillBranch("trigEffWeight", trigWgt[0])
            self.out.fillBranch("trigEffWeightUp", trigWgt[1])
            self.out.fillBranch("trigEffWeightDown", trigWgt[2])

        # L1 prefire weights
        if self._year <= 2017:
            self.out.fillBranch("l1PreFiringWeight", event.L1PreFiringWeight_Nom)
            self.out.fillBranch("l1PreFiringWeightUp", event.L1PreFiringWeight_Up)
            self.out.fillBranch("l1PreFiringWeightDown", event.L1PreFiringWeight_Dn)
        else:
            self.out.fillBranch("l1PreFiringWeight", 1.0)
            self.out.fillBranch("l1PreFiringWeightUp", 1.0)
            self.out.fillBranch("l1PreFiringWeightDown", 1.0)

        # met
        self.out.fillBranch("met", event.met.pt)
        self.out.fillBranch("met_phi", event.met.phi)
        self.out.fillBranch("dphi_met_tkmet", abs(deltaPhi(event.met.phi, event.TkMET_phi)))
        self.out.fillBranch("min_dphi_met_jet", minValue([abs(deltaPhi(event.met, j)) for j in event.ak4jets]))

        # V boson
        self.out.fillBranch("v_pt", event.Vboson.pt())
        self.out.fillBranch("v_eta", event.Vboson.eta())
        self.out.fillBranch("v_phi", event.Vboson.phi())
        self.out.fillBranch("v_mass", event.Vboson.mass())

        # leptons
        self.out.fillBranch("lep1_pt", event.selectedLeptons[0].pt)
        self.out.fillBranch("lep1_eta", event.selectedLeptons[0].eta)
        self.out.fillBranch("lep1_phi", event.selectedLeptons[0].phi)
        self.out.fillBranch("lep1_mass", event.selectedLeptons[0].mass)
        self.out.fillBranch("lep1_pfIso", event.selectedLeptons[0].pfRelIso04_all if abs(
            event.selectedLeptons[0].pdgId) == 13 else event.selectedLeptons[0].pfRelIso03_all)
        self.out.fillBranch("lep1_miniIso", event.selectedLeptons[0].miniPFRelIso_all)
        self.out.fillBranch("lep1_pdgId", event.selectedLeptons[0].pdgId)

        # self.out.fillBranch("lep1_dxy", event.selectedLeptons[0].dxy)
        # self.out.fillBranch("lep1_dxyErr", event.selectedLeptons[0].dxyErr)
        # self.out.fillBranch("lep1_dz", event.selectedLeptons[0].dz)
        # self.out.fillBranch("lep1_dzErr", event.selectedLeptons[0].dzErr)
        # self.out.fillBranch("lep1_ip3d", event.selectedLeptons[0].ip3d)
        # self.out.fillBranch("lep1_sip3d", event.selectedLeptons[0].sip3d)

        if self._channel in ('ZJets', 'TT2L'):
            self.out.fillBranch("lep2_pt", event.selectedLeptons[1].pt)
            self.out.fillBranch("lep2_eta", event.selectedLeptons[1].eta)
            self.out.fillBranch("lep2_phi", event.selectedLeptons[1].phi)
            self.out.fillBranch("lep2_mass", event.selectedLeptons[1].mass)
            self.out.fillBranch("lep2_pdgId", event.selectedLeptons[1].pdgId)

        # event level
        if self._channel in ('WJets', 'TT1L'):
            lep = event.selectedLeptons[0]
            self.out.fillBranch("dphi_lep_met", abs(deltaPhi(lep, event.met)))
            self.out.fillBranch("mt_lep_met", transverseMass(lep, event.met))
        elif self._channel in ('ZJets', 'TT2L'):
            self.out.fillBranch("mt_ll_met", transverseMass(event.Vboson, event.met))

        # AK4 jets, cleaned vs leptons
        self.out.fillBranch("n_btag", len(event.ak4_b_jets))
        self.out.fillBranch("n_ctag", len(event.ak4_c_jets))

        ak4_pt = []
        ak4_eta = []
        ak4_phi = []
        ak4_mass = []
        ak4_tag = []
        ak4_mu_ptfrac = []
        ak4_mu_plus_nem = []
        ak4_mu_pdgId = []
        ak4_hflav = []
        ak4_pflav = []
        ak4_genmatch = []
        ak4_nBHadrons = []
        ak4_nCHadrons = []
        # ak4_bdisc = []
        # ak4_cvbdisc = []
        # ak4_cvldisc = []
        ak4_prob_b = []
        ak4_prob_bb = []
        ak4_prob_c = []
        ak4_prob_cc = []
        ak4_prob_uds = []
        ak4_prob_g = []
        ak4_prob_pu = []
        ak4_prob_undef = []

        for j in event.ak4jets:
            ak4_pt.append(j.pt)
            ak4_eta.append(j.eta)
            ak4_phi.append(j.phi)
            ak4_mass.append(j.mass)
            ak4_tag.append(j.tag)
            ak4_mu_ptfrac.append(j.mu.pt / j.pt if j.mu else 0)
            ak4_mu_plus_nem.append(j.muEF + j.neEmEF)
            ak4_mu_pdgId.append(j.mu.pdgId if j.mu else 0)
            if self.isMC:
                ak4_hflav.append(j.hadronFlavour)
                ak4_pflav.append(j.partonFlavour)
                ak4_genmatch.append(j.genJetIdx >= 0)
                try:
                    ak4_nBHadrons.append(j.nBHadrons)
                    ak4_nCHadrons.append(j.nCHadrons)
                except RuntimeError:
                    ak4_nBHadrons.append(-1)
                    ak4_nCHadrons.append(-1)

            # ak4_bdisc.append(j.btagDeepFlavB)
            # ak4_cvbdisc.append(j.btagDeepFlavCvB)
            # ak4_cvldisc.append(j.btagDeepFlavCvL)
            if self.hasParticleNetAK4:
                ak4_prob_b.append(j.ParticleNetAK4_probb)
                ak4_prob_bb.append(j.ParticleNetAK4_probbb)
                ak4_prob_c.append(j.ParticleNetAK4_probc)
                ak4_prob_cc.append(j.ParticleNetAK4_probcc)
                ak4_prob_uds.append(j.ParticleNetAK4_probuds)
                ak4_prob_g.append(j.ParticleNetAK4_probg)
                ak4_prob_pu.append(j.ParticleNetAK4_probpu)
                ak4_prob_undef.append(j.ParticleNetAK4_probundef)

        self.out.fillBranch("ak4_pt", ak4_pt)
        self.out.fillBranch("ak4_eta", ak4_eta)
        self.out.fillBranch("ak4_phi", ak4_phi)
        self.out.fillBranch("ak4_mass", ak4_mass)
        self.out.fillBranch("ak4_tag", ak4_tag)
        self.out.fillBranch("ak4_mu_ptfrac", ak4_mu_ptfrac)
        self.out.fillBranch("ak4_mu_plus_nem", ak4_mu_plus_nem)
        self.out.fillBranch("ak4_mu_pdgId", ak4_mu_pdgId)
        if self.isMC:
            self.out.fillBranch("ak4_hflav", ak4_hflav)
            self.out.fillBranch("ak4_pflav", ak4_pflav)
            self.out.fillBranch("ak4_genmatch", ak4_genmatch)
            self.out.fillBranch("ak4_nBHadrons", ak4_nBHadrons)
            self.out.fillBranch("ak4_nCHadrons", ak4_nCHadrons)

        self.out.fillBranch("n_mutag", sum(0 < x < 0.5 for x in ak4_mu_ptfrac))

        if self._opts['fillJetTaggingScores']:
            # self.out.fillBranch("ak4_bdisc", ak4_bdisc)
            # self.out.fillBranch("ak4_cvbdisc", ak4_cvbdisc)
            # self.out.fillBranch("ak4_cvldisc", ak4_cvldisc)
            if self.hasParticleNetAK4:
                self.out.fillBranch("ak4_prob_b", ak4_prob_b)
                self.out.fillBranch("ak4_prob_bb", ak4_prob_bb)
                self.out.fillBranch("ak4_prob_c", ak4_prob_c)
                self.out.fillBranch("ak4_prob_cc", ak4_prob_cc)
                self.out.fillBranch("ak4_prob_uds", ak4_prob_uds)
                self.out.fillBranch("ak4_prob_g", ak4_prob_g)
                self.out.fillBranch("ak4_prob_pu", ak4_prob_pu)
                self.out.fillBranch("ak4_prob_undef", ak4_prob_undef)

        self.out.fillBranch("ht", sum([j.pt for j in event.ak4jets]))

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        event.met = METObject(event, "PuppiMET") if self._usePuppiJets else METObject(event, "MET")

        self._selectLeptons(event)
        if self._preSelect(event) is False:
            return False
        if self._selectTriggers(event) is False:
            return False

        self._correctJetAndMET(event)
        self._cleanObjects(event)
        if self._selectEvent(event) is False:
            return False
        # fill
        self._fillEventInfo(event)

        return True


def flavTreeFromConfig():
    import yaml
    with open('flavTree_cfg.json') as f:
        cfg = yaml.safe_load(f)
    return FlavTreeProducer(**cfg)
