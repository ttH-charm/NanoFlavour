import os
import numpy as np
import math
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from ..helpers.utils import deltaPhi, deltaR, deltaR2, deltaEta, closest, polarP4, sumP4, transverseMass, minValue, configLogger, getDigit, closest_pair
from ..helpers.ortHelper import ONNXRuntimeHelperSV
from ..helpers.nnHelper import convert_prob, ensemble
from ..helpers.jetmetCorrector import JetMETCorrector, rndSeed
# from ..helpers.electronCorrector import ElectronScaleResCorrector
# from ..helpers.muonCorrector import MuonScaleResCorrector
from ..helpers.triggerHelper import passTrigger

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

lumi_dict = {2015: 19.52, 2016: 16.81, 2017: 41.48, 2018: 59.83}


class METObject(Object):

    def p4(self):
        return polarP4(self, eta=None, mass=None)


class ZjetsTreeProducer(Module, object):

    def __init__(self, **kwargs):
        self._year = int(kwargs['year'])
        self._jmeSysts = {'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': '',
                          'jer': 'nominal', 'jmr': None, 'met_unclustered': None, 'applyHEMUnc': False,
                          'smearMET': False}
        self._opts = {'use_pn': True, 'min_num_b_or_c_jets': 0,
                      'eval_nn': True, 'eval_mlp': False,
                      'ele_scale': None, 'muon_scale': 'nominal', 'muon_geofit': True,
                      'fillBDTVars': False}
        for k in kwargs:
            if k in self._jmeSysts:
                self._jmeSysts[k] = kwargs[k]
            else:
                self._opts[k] = kwargs[k]
        self._needsJMECorr = any([self._jmeSysts['jec'], self._jmeSysts['jes'],
                                  self._jmeSysts['jer'], self._jmeSysts['jmr'],
                                  self._jmeSysts['met_unclustered'], self._jmeSysts['applyHEMUnc']])

        logger.info('Running for year %s with JME systematics %s, other options %s',
                     str(self._year), str(self._jmeSysts), str(self._opts))

        if self._needsJMECorr:
            self.jetmetCorr = JetMETCorrector(year=self._year, jetType="AK4PFchs", **self._jmeSysts)

        # self.eleCorr = ElectronScaleResCorrector(year=self._year, corr=self._opts['ele_scale'])
        # self.muonCorr = MuonScaleResCorrector(
        #     year=self._year, corr=self._opts['muon_scale'],
        #     geofit=self._opts['muon_geofit'])

        # TODO: update WPs for 2016APV and 2016
        if self._opts['use_pn']:
            # ParticleNetAK4
            self.jetTagWPs = {
                'b': {'expr': 'pn_b',
                      'L': {2015: 0.05, 2016: 0.05, 2017: 0.05, 2018: 0.05}[self._year],
                      'M': {2015: 0.3, 2016: 0.3, 2017: 0.3, 2018: 0.3}[self._year],
                      'T': {2015: 0.8, 2016: 0.8, 2017: 0.8, 2018: 0.8}[self._year],
                      },
                'c': {'expr': 'pn_c',
                      'L': {2015: 0.1, 2016: 0.1, 2017: 0.1, 2018: 0.1}[self._year],
                      'M': {2015: 0.2, 2016: 0.2, 2017: 0.2, 2018: 0.2}[self._year],
                      'T': {2015: 0.5, 2016: 0.5, 2017: 0.5, 2018: 0.5}[self._year],
                      },
                'bc': {'expr': 'pn_sum_bc',
                       'L': {2015: 0.1, 2016: 0.1, 2017: 0.1, 2018: 0.1}[self._year],
                       'M': {2015: 0.2, 2016: 0.2, 2017: 0.2, 2018: 0.2}[self._year],
                       'T': {2015: 0.5, 2016: 0.5, 2017: 0.5, 2018: 0.5}[self._year],
                       },
            }
        else:
            # DeepJet - b-tagging: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
            self.jetTagWPs = {
                'b': {'expr': 'btagDeepFlavB',
                      'L': {2015: 0.0508, 2016: 0.0480, 2017: 0.0532, 2018: 0.0490}[self._year],
                      'M': {2015: 0.2598, 2016: 0.2489, 2017: 0.3040, 2018: 0.2783}[self._year],
                      'T': {2015: 0.6502, 2016: 0.6377, 2017: 0.7476, 2018: 0.7100}[self._year],
                      },
                'c': {'expr': 'btagDeepFlavC',
                      'L': {2015: 0.1, 2016: 0.1, 2017: 0.1, 2018: 0.1}[self._year],
                      'M': {2015: 0.2, 2016: 0.2, 2017: 0.2, 2018: 0.2}[self._year],
                      'T': {2015: 0.4, 2016: 0.4, 2017: 0.4, 2018: 0.4}[self._year],
                      },
                'bc': {'expr': 'btagDeepFlavCvL',
                       'L': {2015: 0.09, 2016: 0.09, 2017: 0.09, 2018: 0.09}[self._year],
                       'M': {2015: 0.14, 2016: 0.14, 2017: 0.14, 2018: 0.14}[self._year],
                       'T': {2015: 0.25, 2016: 0.25, 2017: 0.25, 2018: 0.25}[self._year],
                       },
            }

        # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL
        # NOTE [28.10.2022]: switched to loose PU ID to avoid discontinuity in jet pT distributions
        self.puID_WP = {2015: 1, 2016: 1, 2017: 4, 2018: 4}[self._year]  # L
        # self.puID_WP = {2015: 3, 2016: 3, 2017: 6, 2018: 6}[self._year]  # M
        # self.puID_WP = {2015: 7, 2016: 7, 2017: 7, 2018: 7}[self._year]  # T

        logger.info('Running for year %s with jet tagging WPs %s, jet PU ID WPs %s',
                     str(self._year), str(self.jetTagWPs), str(self.puID_WP))
        
        if self._opts['eval_nn']:
            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data')
            self.nn_helper = ONNXRuntimeHelperSV(
                 preprocess_file = '%s/sv/preprocess.json' % prefix,
                 model_file='%s/sv/model_pn.onnx' % prefix)

    def passJetTag(self, j, category, wp):
        info = self.jetTagWPs[category]
        return getattr(j, info['expr']) > info[wp]

    def beginJob(self):
        if self._needsJMECorr:
            self.jetmetCorr.beginJob()

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        self.hasParticleNetAK4 = 'privateNano' if inputTree.GetBranch(
            'Jet_ParticleNetAK4_probb') else 'jmeNano' if inputTree.GetBranch('Jet_particleNetAK4_B') else None
        if self._opts['use_pn'] and not self.hasParticleNetAK4:
            raise RuntimeError('No ParticleNetAK4 scores in the input NanoAOD!')
        self.rho_branch_name = 'Rho_fixedGridRhoFastjetAll' if bool(
            inputTree.GetBranch('Rho_fixedGridRhoFastjetAll')) else 'fixedGridRhoFastjetAll'

        self.out = wrappedOutputTree

        # NOTE: branch names must start with a lower case letter
        # check keep_and_drop_output.txt
        self.out.branch("year", "I")
        self.out.branch("lumiwgt", "F")

        self.out.branch("passmetfilters", "O")

        # triggers for 2L
        self.out.branch("passTrigElEl", "O")
        #self.out.branch("passTrigElMu", "O")
        self.out.branch("passTrigMuMu", "O")
        # extra single e/mu trigger for 2L
        self.out.branch("passTrig2L_extEl", "O")
        self.out.branch("passTrig2L_extMu", "O")

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
        self.out.branch("lep1_pdgId", "I")
        self.out.branch("lep2_pt", "F")
        self.out.branch("lep2_eta", "F")
        self.out.branch("lep2_phi", "F")
        self.out.branch("lep2_mass", "F")
        self.out.branch("lep2_pdgId", "I")
        
        # event level
        self.out.branch("deltaR_ll", "F")

        # ak4 jets
        self.out.branch("use_pn", "O")
        self.out.branch("n_ak4", "I")
        self.out.branch("n_bl_ak4", "I")
        self.out.branch("n_bm_ak4", "I")
        self.out.branch("n_bt_ak4", "I")

        ## add the SV info: pneet_soft_scores, pT(SV), eta(SV), mass(SV),...
        ## for the first 5
        #self.out.branch("ak4_pt", "F", 20, lenVar="n_ak4")

        if self._opts['eval_nn']:
            for name in self.nn_helper.output_names:
                self.out.branch(name, "F")

        # gen matching
        self.out.branch("n_genTops", "I")
        self.out.branch("tt_category", "I")
        self.out.branch("higgs_decay", "I")

        if self.isMC:
            self.out.branch("genInfo_Nb_top", "I")
            self.out.branch("genInfo_Nc_W", "I")
            self.out.branch("genInfo_Nb_W", "I")
            self.out.branch("genInfo_Nextra_b", "I")
            self.out.branch("genInfo_Nextra_c", "I")
            self.out.branch("genInfo_Nextra_b_hadMult", "I")
            self.out.branch("genInfo_Nextra_c_hadMult", "I")
            self.out.branch("genEventClassifier", "I")

            self.out.branch("h2bb", "O")
            self.out.branch("h2cc", "O")
            self.out.branch("h2tautau", "O")

            self.out.branch("z2bb", "O")
            self.out.branch("z2cc", "O")
            self.out.branch("z2qq", "O")

            self.out.branch("w2cq", "O")
            self.out.branch("w2qq", "O")

            self.out.branch("genH_pt", "F")
            self.out.branch("genZ_pt", "F")
            self.out.branch("genW_pt", "F")

            self.out.branch("hdau_jetidx1", "I")
            self.out.branch("hdau_jetidx2", "I")
            self.out.branch("topb_jetidx1", "I")
            self.out.branch("topb_jetidx2", "I")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def _correctJetAndMET(self, event):
        if self._needsJMECorr:
            rho = getattr(event, self.rho_branch_name)
            # correct AK4 jets and MET
            self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
            self.jetmetCorr.correctJetAndMET(jets=event._allJets, lowPtJets=Collection(event, "CorrT1METJet"),
                                             met=event.met, rawMET=METObject(event, "RawMET"),
                                             defaultMET=METObject(event, "MET"),
                                             rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)  # sort by pt after updating

    def _selectLeptons(self, event):
        # do lepton selection
        event.looseLeptons = []  # used for jet lepton cleaning & lepton counting

        electrons = Collection(event, "Electron")
        for el in electrons:
            el.etaSC = el.eta + el.deltaEtaSC
            if el.pt > 15 and abs(el.eta) < 2.5 and el.mvaFall17V2Iso_WP90:
                event.looseLeptons.append(el)

        muons = Collection(event, "Muon")
        for mu in muons:
            if mu.pt > 15 and abs(mu.eta) < 2.4 and mu.looseId and mu.pfRelIso04_all < 0.25:
                event.looseLeptons.append(mu)

        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)

    def _preSelect(self, event):
        event.selectedLeptons = []  
        if len(event.looseLeptons) < 2:
            return False

        for lep in event.looseLeptons:
            if lep.pt < 20:
                continue
            if abs(lep.pdgId) == 13:
                # mu
                #self.muonCorr.correct(event, lep, self.isMC)
                event.selectedLeptons.append(lep)
            else:
                # el
                #self.eleCorr.correct(event, lep, self.isMC)
                event.selectedLeptons.append(lep)
        if len(event.selectedLeptons) < 2:  return False

        ## select OSSF leptons
        if event.selectedLeptons[0].pdgId + event.selectedLeptons[1].pdgId != 0:
            return False

        ## reconstruct the V(=Z) boson; mass range to be tuned
        Vboson = sumP4(event.selectedLeptons[0], event.selectedLeptons[1])
        if Vboson.M() < 80 or Vboson.M() > 100:
            return False

        return True

    def _cleanObjects(self, event):
        event.ak4jets = []
        for j in event._allJets:
            if not (j.pt > 25 and abs(j.eta) < 2.4 and (j.jetId & 4) and (j.pt > 50 or j.puId >= self.puID_WP)):
                # NOTE: ttH(bb) uses jets w/ pT > 30 GeV, loose PU Id
                # pt, eta, tightIdLepVeto, loose PU ID
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
                j.pn_sum_bc = j.pn_b + j.pn_c
            elif self.hasParticleNetAK4 == 'jmeNano':
                # attach ParticleNet scores
                j.pn_b = j.particleNetAK4_B
                j.pn_c = j.particleNetAK4_B * j.particleNetAK4_CvsB / (
                    1 - j.particleNetAK4_CvsB) if (j.particleNetAK4_CvsB >= 0 and j.particleNetAK4_CvsB < 1) else -1
                j.pn_uds = np.clip(1 - j.pn_b - j.pn_c, 0, 1) * j.particleNetAK4_QvsG if (
                    j.particleNetAK4_QvsG >= 0 and j.particleNetAK4_QvsG < 1) else -1
                j.pn_g = np.clip(1 - j.pn_b - j.pn_c - j.pn_uds, 0, 1) if (
                    j.particleNetAK4_QvsG >= 0 and j.particleNetAK4_QvsG < 1) else -1
                j.pn_sum_bc = j.pn_b + j.pn_c
            # WP-based b- and c-tagging results
            j.btag_L = self.passJetTag(j, 'b', 'L')
            j.btag_M = self.passJetTag(j, 'b', 'M')
            j.btag_T = self.passJetTag(j, 'b', 'T')
            j.ctag_L = self.passJetTag(j, 'c', 'L')
            j.ctag_M = self.passJetTag(j, 'c', 'M')
            j.ctag_T = self.passJetTag(j, 'c', 'T')
            event.ak4jets.append(j)

        event.n_bl_ak4 = []
        event.n_bm_ak4 = []
        event.n_bt_ak4 = []
        

        for jet_idx, j in enumerate(event.ak4jets):
            j.idx = jet_idx
            if self.passJetTag(j, 'b', 'L'):
                event.n_bl_ak4.append(j)
            if self.passJetTag(j, 'b', 'M'):
                event.n_bm_ak4.append(j)
            if self.passJetTag(j, 'b', 'T'):
                event.n_bt_ak4.append(j)

    def _selectEvent(self, event):
        # logger.debug('processing event %d' % event.event)

        event.Vboson = None
        event.Vboson = sumP4(event.selectedLeptons[0], event.selectedLeptons[1])

        if len(event.ak4jets) > 2:
            return False

        # return True if passes selection
        return True


    def _prepSVs(self, event):
        # drop all SVs w/in dR<0.4 of a jet
        event._SVs = []
        for sv in event._allSVs:
            drop = False
            for j in event.ak4jets:
                if deltaR(sv, j) < 0.4:
                    drop = True
            if not drop:
                event._SVs.append(sv)

        # for each SV:  Grab all pfcands w/in dR<0.4
        # create:  [[pf1, pf2, ...]  ]
        event.pfclist = []
        for sv in event._SVs:
            event.pfclist.append([])
            for pf in event._allPFCands:
                if deltaR(sv, pf) < 0.4:
                    event.pfclist[-1].append(pf)


    def _fillGenMatch(self, event):
        if not self.isMC:
            self.out.fillBranch("tt_category", -1)
            self.out.fillBranch("higgs_decay", -1)
            return

        genNb_top = getDigit(event.genTtbarId, 2)
        genNc_W = getDigit(event.genTtbarId, 4)
        genNb_W = getDigit(event.genTtbarId, 3)  # for BSM models giving W -> b decays
        genNextra_b = 0
        genNextra_b = 0
        genNextra_c = 0
        genNextra_b_hadMult = 0
        genNextra_c_hadMult = 0
        if getDigit(event.genTtbarId, 1) == 5:
            genNextra_b = 2 if getDigit(event.genTtbarId, 0) > 2 else 1
            genNextra_b_hadMult = getDigit(event.genTtbarId, 0)
        elif getDigit(event.genTtbarId, 1) == 4:
            genNextra_c = 2 if getDigit(event.genTtbarId, 0) > 2 else 1
            genNextra_c_hadMult = getDigit(event.genTtbarId, 0)

        self.out.fillBranch("genInfo_Nb_top", genNb_top)
        self.out.fillBranch("genInfo_Nc_W", genNc_W)
        self.out.fillBranch("genInfo_Nb_W", genNb_W)
        self.out.fillBranch("genInfo_Nextra_b", genNextra_b)
        self.out.fillBranch("genInfo_Nextra_c", genNextra_c)
        self.out.fillBranch("genInfo_Nextra_b_hadMult", genNextra_b_hadMult)
        self.out.fillBranch("genInfo_Nextra_c_hadMult", genNextra_c_hadMult)

        try:
            genparts = event.genparts
        except RuntimeError as e:
            genparts = Collection(event, "GenPart")
            for idx, gp in enumerate(genparts):
                if 'dauIdx' not in gp.__dict__:
                    gp.dauIdx = []
                if gp.genPartIdxMother >= 0:
                    mom = genparts[gp.genPartIdxMother]
                    if 'dauIdx' not in mom.__dict__:
                        mom.dauIdx = [idx]
                    else:
                        mom.dauIdx.append(idx)
            event.genparts = genparts

        def isHadronic(gp):
            if len(gp.dauIdx) == 0:
                logger.warning(f'Particle (pdgId={gp.pdgId}) has no daughters!')
            for idx in gp.dauIdx:
                if abs(genparts[idx].pdgId) < 6:
                    return True
            return False

        def getFinal(gp):
            for idx in gp.dauIdx:
                dau = genparts[idx]
                if dau.pdgId == gp.pdgId:
                    return getFinal(dau)
            return gp

        nGenTops = 0
        nGenWs = 0
        nGenZs = 0
        nGenHs = 0

        lepGenTops = []
        hadGenTops = []
        bFromTops = []
        hadGenWs = []
        hadGenZs = []
        hadGenHs = []
        lepGenHs = []

        for gp in genparts:
            if gp.statusFlags & (1 << 13) == 0:
                continue
            if abs(gp.pdgId) == 6:
                nGenTops += 1
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if abs(dau.pdgId) == 24:
                        genW = getFinal(dau)
                        gp.genW = genW
                        if isHadronic(genW):
                            hadGenTops.append(gp)
                        else:
                            lepGenTops.append(gp)
                    elif abs(dau.pdgId) in (1, 3, 5):
                        gp.genB = dau
                        bFromTops.append(dau)
            elif abs(gp.pdgId) == 24:
                nGenWs += 1
                if isHadronic(gp):
                    hadGenWs.append(gp)
            elif abs(gp.pdgId) == 23:
                nGenZs += 1
                if isHadronic(gp):
                    hadGenZs.append(gp)
            elif abs(gp.pdgId) == 25:
                nGenHs += 1
                if isHadronic(gp):
                    hadGenHs.append(gp)
                else:
                    lepGenHs.append(gp)

        def get_daughters(parton):
            try:
                if abs(parton.pdgId) == 6:
                    return (parton.genB, genparts[parton.genW.dauIdx[0]], genparts[parton.genW.dauIdx[1]])
                elif abs(parton.pdgId) in (23, 24, 25):
                    return (genparts[parton.dauIdx[0]], genparts[parton.dauIdx[1]])
            except IndexError:
                logger.warning(f'Failed to get daughter for particle (pdgId={parton.pdgId})!')
                return tuple()

        genH = hadGenHs[0] if len(hadGenHs) else None
        # 0: no hadronic H; else: max pdgId of the daughters
        hdecay_ = abs(get_daughters(genH)[0].pdgId) if genH else 0

        genHlep = lepGenHs[0] if len(lepGenHs) else None
        # 0: no hadronic H; else: max pdgId of the daughters
        hdecaylep_ = abs(get_daughters(genHlep)[0].pdgId) if genHlep else 0

        genZ = hadGenZs[0] if len(hadGenZs) else None
        # 0: no hadronic Z; else: max pdgId of the daughters
        zdecay_ = abs(get_daughters(genZ)[0].pdgId) if genZ else 0

        genW = hadGenWs[0] if len(hadGenWs) else None
        # 0: no hadronic W; else: max pdgId of the daughters
        wdecay_ = max([abs(d.pdgId) for d in get_daughters(genW)], default=0) if genW else 0

        self.out.fillBranch("n_genTops", nGenTops)
        self.out.fillBranch("tt_category", getDigit(event.genTtbarId, 1))
        self.out.fillBranch("higgs_decay", hdecay_ if hdecay_ != 0 else hdecaylep_)

        self.out.fillBranch("h2bb", hdecay_ == 5)
        self.out.fillBranch("h2cc", hdecay_ == 4)
        self.out.fillBranch("h2tautau", hdecaylep_ == 15)

        self.out.fillBranch("z2bb", zdecay_ == 5)
        self.out.fillBranch("z2cc", zdecay_ == 4)
        self.out.fillBranch("z2qq", zdecay_ > 0 and zdecay_ < 4)

        self.out.fillBranch("w2cq", wdecay_ == 4)
        self.out.fillBranch("w2qq", wdecay_ > 0 and wdecay_ < 4)

        self.out.fillBranch("genH_pt", genH.pt if genH else -1)
        self.out.fillBranch("genZ_pt", genZ.pt if genZ else -1)
        self.out.fillBranch("genW_pt", genW.pt if genW else -1)

        # jet-parton matching
        hdau_jetidx = [-1, -1]
        if genH:
            for idx, dau in enumerate(get_daughters(genH)):
                if idx >= 2:
                    break
                for j in event.ak4jets:
                    if deltaR(j, dau) < 0.4:
                        hdau_jetidx[idx] = j.idx
                        break
        self.out.fillBranch("hdau_jetidx1", hdau_jetidx[0])
        self.out.fillBranch("hdau_jetidx2", hdau_jetidx[1])

        top_bjetidx = [-1, -1]
        for idx, dau in enumerate(bFromTops):
            if idx >= 2:
                break
            for j in event.ak4jets:
                if deltaR(j, dau) < 0.4:
                    top_bjetidx[idx] = j.idx
                    break
        self.out.fillBranch("topb_jetidx1", top_bjetidx[0])
        self.out.fillBranch("topb_jetidx2", top_bjetidx[1])

        genEventClassifier_id = -99
        if nGenHs > 0:  # there is a gen level Higgs
            if hdecay_ == 4:
                genEventClassifier_id = 0  # higgs to cc
            elif hdecay_ == 5:
                genEventClassifier_id = 1  # higgs to bb
            elif hdecaylep_ == 15:
                genEventClassifier_id = 2  # higgs to tautau
        else:  # tt categories
            if getDigit(event.genTtbarId, 1) == 0:  # no HFs accompanying tt
                genEventClassifier_id = 3  # tt+LF
            if getDigit(event.genTtbarId, 1) == 4:  # c's accompanying tt
                if getDigit(event.genTtbarId, 0) == 1:  # tt+c
                    genEventClassifier_id = 4
                if getDigit(event.genTtbarId, 0) == 2:  # tt+2c, 1 c jet with 2 hadrons
                    genEventClassifier_id = 5
                if getDigit(event.genTtbarId, 0) > 2:  # tt+cc
                    genEventClassifier_id = 6
            if getDigit(event.genTtbarId, 1) == 5:  # b's abbompanying tt
                if getDigit(event.genTtbarId, 0) == 1:  # tt+b
                    genEventClassifier_id = 7
                if getDigit(event.genTtbarId, 0) == 2:  # tt+2b, 1 b jet with 2 hadrons
                    genEventClassifier_id = 8
                if getDigit(event.genTtbarId, 0) > 2:  # tt+bb
                    genEventClassifier_id = 9

        self.out.fillBranch("genEventClassifier", genEventClassifier_id)

    def _fillEventInfo(self, event):
        self.out.fillBranch("year", self._year)
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

        # trigger
        # !!! NOTE: make sure to update `keep_and_drop_input.txt` !!!
        if self._year <= 2016:
            self.out.fillBranch("passTrigElEl", passTrigger(event, 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'))
            #self.out.fillBranch("passTrigElMu", passTrigger(event,
            #                                                ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
            #                                                 'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
            #                                                 'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL',
            #                                                 'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ']))
            self.out.fillBranch(
                "passTrigMuMu",
                passTrigger(event, ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL'])
                if event.run <= 280385 else  # Run2016G
                passTrigger(event, ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ']))
            self.out.fillBranch("passTrig2L_extEl", passTrigger(event, 'HLT_Ele27_WPTight_Gsf'))
            self.out.fillBranch("passTrig2L_extMu", passTrigger(event, ['HLT_IsoMu24', 'HLT_IsoTkMu24']))
        elif self._year == 2017:
            self.out.fillBranch("passTrigElEl", passTrigger(event, ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                                    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ']))
            #self.out.fillBranch("passTrigElMu", passTrigger(event,
            #                                                ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
            #                                                 'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
            #                                                 'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
            #                                                 'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ']))
            self.out.fillBranch(
                "passTrigMuMu", passTrigger(event, 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ')
                if event.run <= 299329 else  # Run2017B
                passTrigger(event, 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8'))
            self.out.fillBranch("passTrig2L_extEl", passTrigger(event, 'HLT_Ele32_WPTight_Gsf_L1DoubleEG'))
            self.out.fillBranch("passTrig2L_extMu", passTrigger(event, ['HLT_IsoMu24_eta2p1', 'HLT_IsoMu27']))
        elif self._year == 2018:
            self.out.fillBranch("passTrigElEl", passTrigger(event,
                                                            ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                             'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ']))
            #self.out.fillBranch("passTrigElMu", passTrigger(event,
            #                                                ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
            #                                                 'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
            #                                                 'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
            #                                                 'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ']))
            self.out.fillBranch("passTrigMuMu", passTrigger(event,
                                                            ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8',
                                                             'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8']))
            self.out.fillBranch("passTrig2L_extEl", passTrigger(event, 'HLT_Ele32_WPTight_Gsf'))
            self.out.fillBranch("passTrig2L_extMu", passTrigger(event, 'HLT_IsoMu24'))

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
        self.out.fillBranch("v_pt", event.Vboson.Pt())
        self.out.fillBranch("v_eta", event.Vboson.Eta())
        self.out.fillBranch("v_phi", event.Vboson.Phi())
        self.out.fillBranch("v_mass", event.Vboson.M())

        # leptons
        self.out.fillBranch("lep1_pt", event.selectedLeptons[0].pt)
        self.out.fillBranch("lep1_eta", event.selectedLeptons[0].eta)
        self.out.fillBranch("lep1_phi", event.selectedLeptons[0].phi)
        self.out.fillBranch("lep1_mass", event.selectedLeptons[0].mass)
        self.out.fillBranch("lep1_pdgId", event.selectedLeptons[0].pdgId)
        self.out.fillBranch("lep2_pt", event.selectedLeptons[1].pt)
        self.out.fillBranch("lep2_eta", event.selectedLeptons[1].eta)
        self.out.fillBranch("lep2_phi", event.selectedLeptons[1].phi)
        self.out.fillBranch("lep2_mass", event.selectedLeptons[1].mass)
        self.out.fillBranch("lep2_pdgId", event.selectedLeptons[1].pdgId)

        # event level
        self.out.fillBranch("deltaR_ll", deltaR(event.selectedLeptons[0], event.selectedLeptons[1]))

        # AK4 jets, cleaned vs leptons
        self.out.fillBranch("use_pn", self._opts['use_pn'])
        self.out.fillBranch("n_ak4", len(event.ak4jets))
        self.out.fillBranch("n_bl_ak4", len(event.n_bl_ak4))
        self.out.fillBranch("n_bm_ak4", len(event.n_bm_ak4))
        self.out.fillBranch("n_bt_ak4", len(event.n_bt_ak4))

    
    def _fillNN(self, event):
        # ====== fill PN/nn vars ======
        for i in range(len(event._SVs)):
            inputs = {
                'pfcand_pt_log_nopuppi': [],
                'pfcand_e_log_nopuppi': [],
                'pfcand_etarel': [],
                'pfcand_phirel': [],
                'pfcand_abseta': [],
                'pfcand_charge': [],
                'pfcand_VTX_ass': [],
                'pfcand_lostInnerHits': [],
                'pfcand_normchi2': [],
                'pfcand_quality': [],
                'pfcand_dz': [],
                'pfcand_dzsig': [],
                'pfcand_dxy': [],
                'pfcand_dxysig': [],
                'pfcand_btagEtaRel': [],
                'pfcand_btagPtRatio': [],
                'pfcand_btagPParRatio': [],
                'pfcand_btagSip3dVal': [],
                'pfcand_btagSip3dSig': [],
                'pfcand_btagJetDistVal': [],
                'pfcand_puppiw': [],
                'pfcand_mask': [],
            }


            sv = event._SVs[i]
            if len(event.pfclist[i])==0:
                continue  # ignore SVs w/ no matched pfcands

            for pfc in event.pfclist[i]:
                vec = ROOT.TLorentzVector()
                vec.SetPtEtaPhiM(pfc.pt, pfc.eta, pfc.phi, pfc.mass)
                inputs['pfcand_pt_log_nopuppi'].append(np.log(pfc.pt))
                inputs['pfcand_e_log_nopuppi'].append(np.log(vec.E()))
                inputs['pfcand_etarel'].append(pfc.eta - sv.eta)
                inputs['pfcand_phirel'].append(pfc.phi - sv.eta)
                inputs['pfcand_abseta'].append(abs(pfc.eta))
                inputs['pfcand_charge'].append(pfc.charge)
                inputs['pfcand_VTX_ass'].append(pfc.pvAssocQuality)
                inputs['pfcand_lostInnerHits'].append(pfc.lostInnerHits)
                inputs['pfcand_normchi2'].append(np.floor(pfc.trkChi2 if pfc.trkChi2 != -1 else 999))#
                inputs['pfcand_quality'].append(pfc.trkQuality)
                inputs['pfcand_dz'].append(pfc.dz)
                inputs['pfcand_dzsig'].append(pfc.dz/pfc.dzErr)
                inputs['pfcand_dxy'].append(pfc.d0)
                inputs['pfcand_dxysig'].append(pfc.d0/pfc.d0Err)
                inputs['pfcand_btagEtaRel'].append(pfc.btagEtaRel)
                inputs['pfcand_btagPtRatio'].append(pfc.btagPtRatio)
                inputs['pfcand_btagPParRatio'].append(pfc.btagPParRatio)
                inputs['pfcand_btagSip3dVal'].append(pfc.btagSip3dVal)
                inputs['pfcand_btagSip3dSig'].append(pfc.btagSip3dSig)
                inputs['pfcand_btagJetDistVal'].append(pfc.btagJetDistVal)
                inputs['pfcand_puppiw'].append(pfc.puppiWeight)
                inputs['pfcand_mask'].append(1)


            # TODO: must fill for each sv
            outputs = self.nn_helper.predict(inputs) #, model_idx)
            for name in outputs:
                self.out.fillBranch(name, outputs[name])
    


    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        # NOTE:  For SV tagging, need to call fill/etc once for each

        event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        event.met = METObject(event, "MET")
        # NEW for PN
        event._allPFCands = Collection(event, "PFCands")
        event._allSVs = Collection(event, "SV")

        self._selectLeptons(event)
        if self._preSelect(event) is False:
            return False
        self._correctJetAndMET(event)
        self._cleanObjects(event)  # checked
        if self._selectEvent(event) is False:  # checked
            return False
        # NEW:  create pfclist of pfc's matched to each SV: [[pfc1, pfc2...], ...]
        self._prepSVs(event)

        # fill
        self._fillEventInfo(event)
        self._fillGenMatch(event)

        if self._opts['eval_nn']:
            self._fillNN(event)

        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def zjetsTree_2016(): return ZjetsTreeProducer(year=2016)
def zjetsTree_2017(): return ZjetsTreeProducer(year=2017)
def zjetsTree_2018(): return ZjetsTreeProducer(year=2018)

def zjetsTreeFromConfig():
    import yaml
    with open('zjets_cfg.json') as f:
        cfg = yaml.safe_load(f)
    return ZjetsTreeProducer(**cfg)
