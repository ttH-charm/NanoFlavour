import ROOT
import os
import math
import yaml
import json
import numpy as np
from functools import partial
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection


def debug(msg, activated=False):
    if activated:
        print(' > ', msg)


def get_sf_json(sf_dict, var1, var2, inflate_unc_out_bounds=True, no_unc=False):

    # `eta` refers to the first binning var, `pt` refers to the second binning var
    eta, pt = var1, var2

    # if no bin is found, search for closest one, and double the uncertainty
    closestEtaBin = None
    closestEta = 9999.
    etaFound = False
    for etaKey in sf_dict:
        etaL, etaH = [float(v) for v in etaKey.split(':')[1].replace(' ', '').strip('[]').split(',')]

        if abs(etaL - eta) < closestEta or abs(etaH - eta) < closestEta and not etaFound:
            closestEta = min(abs(etaL - eta), abs(etaH - eta))
            closestEtaBin = etaKey

        if (eta > etaL and eta < etaH):
            closestEtaBin = etaKey
            etaFound = True
            break

    if closestEtaBin is None:
        return np.ones(3)

    closestPtBin = None
    closestPt = 9999.
    ptFound = False
    for ptKey in sf_dict[closestEtaBin]:
        ptL, ptH = [float(v) for v in ptKey.split(':')[1].replace(' ', '').strip('[]').split(',')]

        if abs(ptL - pt) < closestPt or abs(ptH - pt) < closestPt and not ptFound:
            closestPt = min(abs(ptL - pt), abs(ptH - pt))
            closestPtBin = ptKey

        if (pt > ptL and pt < ptH):
            closestPtBin = ptKey
            ptFound = True
            break

    if closestPtBin is None:
        return np.ones(3)

    result = sf_dict[closestEtaBin][closestPtBin]
    val = result['value']
    if no_unc:
        return np.array([val, val, val])

    err = result['error'] if 'syst' not in result else math.hypot(
        result['stat'] if 'stat' in result else result['error'], result['syst'])
    if not etaFound or not ptFound:
        if inflate_unc_out_bounds:
            err *= 2.

    return np.array([val, val + err, val - err])


def get_sf_root(hist, x, y, inflate_unc_out_bounds=True, no_unc=False):
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

    def __init__(self, prefix, file=None, key=None):
        if file is None:
            self.obj = None
            return

        self.vars = key.split('/')[-1] if '/' in key else None
        filepath = os.path.join(prefix, file)
        if filepath.endswith('.root'):
            self._rtfile = ROOT.TFile(filepath)
            self.obj = self._rtfile.Get(key)
        elif filepath.endswith('.json'):
            with open(filepath) as f:
                self.obj = json.load(f)
                for k in key.split('/'):
                    self.obj = self.obj[k]
        else:
            raise RuntimeError(
                'Invalid file %s. Only root or json files are supported.' % filepath)

    def __call__(self, x, y, inflate_unc_out_bounds=True, no_unc=False):
        if self.obj is None:
            return np.ones(3)
        if isinstance(self.obj, dict):
            return get_sf_json(self.obj, x, y, inflate_unc_out_bounds, no_unc)
        else:
            return get_sf_root(self.obj, x, y, inflate_unc_out_bounds, no_unc)


class ElectronSFProducer(Module):

    def __init__(self, year, channel):
        self.year = int(year)
        self.channel = channel
        basedir = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data/leptonSF/%s' % year)
        with open(os.path.join(basedir, 'lepton_info_%s.yaml' % year)) as f:
            self.cfg = yaml.safe_load(f)['electron'][channel]
        print('Init electron SF:', self.cfg)
        self.SFs = {k: SFHelper(prefix=basedir, **self.cfg[k]) for k in self.cfg}

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        if self.isMC:
            self.out = wrappedOutputTree

            self.out.branch('elEffWeight', "F")
            self.out.branch('elEffWeight_UP', "F")
            self.out.branch('elEffWeight_DOWN', "F")

    def getSF_0L(self, event):

        eventWgt = np.ones(3)

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

        genLeps = []
        for gp in genparts:
            if gp.status == 1 and (gp.statusFlags & (1 << 0)) and abs(gp.pdgId) == 11:
                # statusFlags bit 0: isPrompt
                if gp.pt > 15 and abs(gp.eta) < 2.5:
                    genLeps.append(gp)

        for idx, lep in enumerate(genLeps):
            debug('gen electron %d, pt:%.1f, eta:%.2f' % (idx, lep.pt, lep.eta))

            mcEff = self.SFs['effMC'](lep.eta, lep.pt, no_unc=True)
            debug('ele MC eff: %s' % str(mcEff))

            sfReco = self.SFs['RECO'](lep.eta, lep.pt, False)
            debug('ele Reco SF: %s' % str(sfReco))

            sfIdIso = self.SFs['ID'](lep.eta, lep.pt)
            debug('ele ID-ISO SF: %s' % str(sfIdIso))

            vetoSF = np.clip((1 - sfReco * sfIdIso * mcEff), 1e-3, 1) / np.clip((1 - mcEff), 1e-3, 1)
            debug('ele veto SF: %s' % str(vetoSF))

            eventWgt *= vetoSF

        eventWgt = np.clip(eventWgt, 0, 2)
        debug('el Full SF: %s' % str(eventWgt))
        return eventWgt

    def getSF_1L(self, event):

        if len(event.selectedLeptons) != 1 or abs(event.selectedLeptons[0].pdgId) != 11:
            return np.ones(3)

        lep = event.selectedLeptons[0]
        debug('electron pt:%.1f, eta:%.2f, etaSC:%.2f' % (lep.pt, lep.eta, lep.etaSC))

        sfReco = self.SFs['RECO'](lep.etaSC, lep.pt, False)
        debug('ele Reco SF: %s' % str(sfReco))

        sfIdIso = self.SFs['ID'](lep.etaSC, lep.pt)
        debug('ele ID-ISO SF: %s' % str(sfIdIso))

        sfTrig = self.SFs['Trig'](lep.etaSC, lep.pt)
        debug('ele Trig SF: %s' % str(sfTrig))

        eventWgt = sfReco * sfIdIso * sfTrig
        debug('el Full SF: %s' % str(eventWgt))
        return eventWgt

    def getSF_2L(self, event):
        if len(event.selectedLeptons) < 2 or abs(event.selectedLeptons[0].pdgId) != 11:
            return np.ones(3)

        eventWgt = np.ones(3)

        for idx, lep in enumerate(event.selectedLeptons[:2]):
            debug('electron %d, pt:%.1f, eta:%.2f, etaSC:%.2f' % (idx, lep.pt, lep.eta, lep.etaSC))

            sfReco = self.SFs['RECO'](lep.etaSC, lep.pt, False)
            debug('ele Reco SF: %s' % str(sfReco))

            sfIdIso = self.SFs['ID'](lep.etaSC, lep.pt)
            debug('ele ID-ISO SF: %s' % str(sfIdIso))

            eventWgt = eventWgt * sfReco * sfIdIso

        def trigSF(mc1, mc2, sf1, sf2):
            mc_eff = mc1 + mc2 - mc1 * mc2
            data_eff = (sf1 * mc1) + (sf2 * mc2) - (sf1 * mc1) * (sf2 * mc2)
            debug('ele1 trig eff MC: %s' % str(mc1))
            debug('ele2 trig eff MC: %s' % str(mc2))
            debug('ele1 trig sf: %s' % str(sf1))
            debug('ele2 trig sf: %s' % str(sf2))
            debug(' trig eff MC: %s' % str(mc_eff))
            debug(' trig eff data: %s' % str(data_eff))
            return data_eff / mc_eff

        eventWgt *= trigSF(
            self.SFs['TrigMC'](event.selectedLeptons[0].etaSC, event.selectedLeptons[0].pt, no_unc=True),
            self.SFs['TrigMC'](event.selectedLeptons[1].etaSC, event.selectedLeptons[1].pt, no_unc=True),
            self.SFs['Trig'](event.selectedLeptons[0].etaSC, event.selectedLeptons[0].pt),
            self.SFs['Trig'](event.selectedLeptons[1].etaSC, event.selectedLeptons[1].pt),
        )

        debug('el Full SF: %s' % str(eventWgt))
        return eventWgt

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        if not self.isMC:
            return True

        sf = np.ones(3)

        if self.channel == '0L':
            sf = self.getSF_0L(event)
        elif self.channel == '1L':
            sf = self.getSF_1L(event)
        elif self.channel == '2L':
            sf = self.getSF_2L(event)

        self.out.fillBranch('elEffWeight', sf[0])
        self.out.fillBranch('elEffWeight_UP', sf[1])
        self.out.fillBranch('elEffWeight_DOWN', sf[2])

        return True


class MuonSFProducer(Module):

    def __init__(self, year, channel):
        self.year = int(year)
        self.channel = channel
        basedir = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoFlavour/data/leptonSF/%s' % year)
        with open(os.path.join(basedir, 'lepton_info_%s.yaml' % year)) as f:
            self.cfg = yaml.safe_load(f)['muon'][channel]
        print('Init muon SF:', self.cfg)
        self.SFs = {k: SFHelper(prefix=basedir, **self.cfg[k]) for k in self.cfg}

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        if self.isMC:
            self.out = wrappedOutputTree

            self.out.branch('muEffWeight', "F")
            self.out.branch('muEffWeight_UP', "F")
            self.out.branch('muEffWeight_DOWN', "F")

    def get(self, sf_type, lep):
        obj = self.SFs[sf_type]
        if obj.vars == 'abseta_pt' or 'abseta_pt_ratio':
            sf = obj(abs(lep.eta), lep.pt)
        elif obj.vars == 'pt_eta':
            sf = obj(lep.pt, lep.eta)
        else:
            raise RuntimeError('Invalid variable ordering %s' % obj.vars)
        debug('mu %s SF: %s' % (sf_type, str(sf)))
        return sf

    def get_avg(self, sf_type, lep, weights={'BCDEF': 0.55, 'GH': 0.45}):
        sf = np.zeros(3)
        for k in weights:
            sf += weights[k] * self.get(sf_type + '_' + k, lep)
        debug('mu avg %s SF: %s' % (sf_type, str(sf)))
        return sf

    def getSF_0L(self, event):
        eventWgt = np.ones(3)

        get = self.get_avg if self.year == 2016 else self.get

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

        genLeps = []
        for gp in genparts:
            if gp.status == 1 and (gp.statusFlags & (1 << 0)) and abs(gp.pdgId) == 13:
                # statusFlags bit 0: isPrompt
                if gp.pt > 15 and abs(gp.eta) < 2.4:
                    genLeps.append(gp)

        for idx, lep in enumerate(genLeps):
            debug('gen muon %d, pt:%.1f, eta:%.2f' % (idx, lep.pt, lep.eta))

            mcEff = self.SFs['effMC'](lep.eta, lep.pt, no_unc=True)
            debug('mu MC eff: %s' % str(mcEff))

            sfID = get('ID', lep)
            debug('mu ID SF: %s' % str(sfID))

            sfISO = get('ISO', lep)
            debug('mu ISO SF: %s' % str(sfISO))

            vetoSF = np.clip((1 - sfID * sfISO * mcEff), 1e-3, 1) / np.clip((1 - mcEff), 1e-3, 1)
            debug('mu veto SF: %s' % str(vetoSF))

            eventWgt *= vetoSF

        eventWgt = np.clip(eventWgt, 0, 2)
        debug('mu Full SF: %s' % str(eventWgt))
        return eventWgt

    def getSF_1L(self, event):

        if len(event.selectedLeptons) != 1 or abs(event.selectedLeptons[0].pdgId) != 13:
            return np.ones(3)

        lep = event.selectedLeptons[0]
        debug('muon pt:%.1f, eta:%.2f' % (lep.pt, lep.eta))

        get = self.get_avg if self.year == 2016 else self.get

        sfID = get('ID', lep)
        sfISO = get('ISO', lep)
        sfTrig = get('Trig', lep)

        eventWgt = sfID * sfISO * sfTrig
        debug('mu Full SF: %s' % str(eventWgt))
        return eventWgt

    def getSF_2L(self, event):
        if len(event.selectedLeptons) < 2 or abs(event.selectedLeptons[0].pdgId) != 13:
            return np.ones(3)

        eventWgt = np.ones(3)

        get = self.get_avg if self.year == 2016 else self.get
        getTrig = partial(self.get_avg, weights={'Before': 0.15, 'After': 0.85}) if self.year == 2018 else get
        for idx, lep in enumerate(event.selectedLeptons[:2]):
            debug('muon %d, pt:%.1f, eta:%.2f' % (idx, lep.pt, lep.eta))

            sfID = get('ID', lep)
            sfISO = get('ISO', lep)

            eventWgt = eventWgt * sfID * sfISO

        def trigSF(mc1, mc2, sf1, sf2):
            mc_eff = mc1 + mc2 - mc1 * mc2
            data_eff = (sf1 * mc1) + (sf2 * mc2) - (sf1 * mc1) * (sf2 * mc2)
            debug('mu1 trig eff MC: %s' % str(mc1))
            debug('mu2 trig eff MC: %s' % str(mc2))
            debug('mu1 trig sf: %s' % str(sf1))
            debug('mu2 trig sf: %s' % str(sf2))
            debug(' trig eff MC: %s' % str(mc_eff))
            debug(' trig eff data: %s' % str(data_eff))
            return data_eff / mc_eff

        eventWgt *= trigSF(
            self.SFs['TrigMC'](event.selectedLeptons[0].eta, event.selectedLeptons[0].pt, no_unc=True),
            self.SFs['TrigMC'](event.selectedLeptons[1].eta, event.selectedLeptons[1].pt, no_unc=True),
            getTrig('Trig', event.selectedLeptons[0]),
            getTrig('Trig', event.selectedLeptons[1]),
        )

        debug('mu Full SF: %s' % str(eventWgt))
        return eventWgt

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        if not self.isMC:
            return True

        sf = np.ones(3)

        if self.channel == '0L':
            sf = self.getSF_0L(event)
        elif self.channel == '1L':
            sf = self.getSF_1L(event)
        elif self.channel == '2L':
            sf = self.getSF_2L(event)

        self.out.fillBranch('muEffWeight', sf[0])
        self.out.fillBranch('muEffWeight_UP', sf[1])
        self.out.fillBranch('muEffWeight_DOWN', sf[2])

        return True

# ------------ 2016 ------------------------


def electronSF_2016_0L():
    return ElectronSFProducer(year=2016, channel='0L')


def electronSF_2016_1L():
    return ElectronSFProducer(year=2016, channel='1L')


def electronSF_2016_2L():
    return ElectronSFProducer(year=2016, channel='2L')


def muonSF_2016_0L():
    return MuonSFProducer(year=2016, channel='0L')


def muonSF_2016_1L():
    return MuonSFProducer(year=2016, channel='1L')


def muonSF_2016_2L():
    return MuonSFProducer(year=2016, channel='2L')


# ------------ 2017 ------------------------
def electronSF_2017_0L():
    return ElectronSFProducer(year=2017, channel='0L')


def electronSF_2017_1L():
    return ElectronSFProducer(year=2017, channel='1L')


def electronSF_2017_2L():
    return ElectronSFProducer(year=2017, channel='2L')


def muonSF_2017_0L():
    return MuonSFProducer(year=2017, channel='0L')


def muonSF_2017_1L():
    return MuonSFProducer(year=2017, channel='1L')


def muonSF_2017_2L():
    return MuonSFProducer(year=2017, channel='2L')


# ------------ 2018 ------------------------
def electronSF_2018_0L():
    return ElectronSFProducer(year=2018, channel='0L')


def electronSF_2018_1L():
    return ElectronSFProducer(year=2018, channel='1L')


def electronSF_2018_2L():
    return ElectronSFProducer(year=2018, channel='2L')


def muonSF_2018_0L():
    return MuonSFProducer(year=2018, channel='0L')


def muonSF_2018_1L():
    return MuonSFProducer(year=2018, channel='1L')


def muonSF_2018_2L():
    return MuonSFProducer(year=2018, channel='2L')
