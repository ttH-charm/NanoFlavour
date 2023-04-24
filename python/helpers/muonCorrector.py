from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
import os
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True


def mk_safe(fct, *args):
    try:
        return fct(*args)
    except Exception as e:
        if any('Error in function boost::math::erf_inv' in arg
               for arg in e.args):
            print(
                'WARNING: catching exception and returning -1. Exception arguments: %s'
                % e.args)
            return -1.
        else:
            raise e


GEOFIT_C = '''
// https://github.com/UFLX2MuMu/GeoFit
// Piece of code for applying GeoFit corrections to muon pT in H->mumu analysis
// Input should be d0_BS*muon charge, pt_Roch for the muon, muon eta and the year of the data/MC sample
// The output is the corrected muon pT.
namespace GeoFit{
  float PtCorrGeoFit(float d0_BS_charge, float pt_Roch, float eta, int year) {
    float pt_cor = 0.0;
    if (year == 2016) {
      if      (abs(eta) < 0.9) pt_cor = 411.34 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
      else if (abs(eta) < 1.7) pt_cor = 673.40 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
      else                     pt_cor = 1099.0 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
    }
    else if (year == 2017) {
      if      (abs(eta) < 0.9) pt_cor = 582.32 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
      else if (abs(eta) < 1.7) pt_cor = 974.05 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
      else                     pt_cor = 1263.4 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
    }
    else if (year == 2018) {
      if      (abs(eta) < 0.9) pt_cor = 650.84 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
      else if (abs(eta) < 1.7) pt_cor = 988.37 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
      else                     pt_cor = 1484.6 * d0_BS_charge * pt_Roch * pt_Roch / 10000.0;
    }
    return (pt_Roch - pt_cor);
  } // end of float PtCorrGeoFit(float d0_BS_charge, float pt_Roch, float eta, int year)
} // end namespace GeoFit
'''


def rndSeed(event, obj):
    seed = (event.run << 20) + (event.luminosityBlock << 10) + event.event + int(obj.eta / 0.01)
    return seed


class MuonScaleResCorrector(object):
    def __init__(self, year, corr='nominal', geofit=True):
        self.year = year
        self.corr = corr
        self.geofit = geofit
        if self.corr:
            for library in ["libPhysicsToolsNanoFlavour"]:
                if library not in ROOT.gSystem.GetLibraries():
                    print("Load Library '%s'" % library.replace("lib", ""))
                    ROOT.gSystem.Load(library)
            fn = os.environ['CMSSW_BASE'] + '/src/PhysicsTools/NanoAODTools/python/postprocessing/data/roccor.Run2.v3/RoccoR%s.txt' % year
            print('Loading muon scale and smearing file from %s' % fn)
            self._roccor = ROOT.RoccoR(fn)
            self.rnd = ROOT.TRandom3()

            if self.geofit:
                ROOT.gROOT.ProcessLine(GEOFIT_C)
                self._geofit = ROOT.GeoFit.PtCorrGeoFit

    def correct(self, event, mu, is_mc):
        if not self.corr:
            return

        roccor = self._roccor
        if is_mc:
            try:
                genparticles = event._genparticles
            except RuntimeError:
                genparticles = Collection(event, "GenPart")
                event._genparticles = genparticles

            if mu.genPartIdx >= 0 and mu.genPartIdx < len(genparticles):
                genMu = genparticles[mu.genPartIdx]
                pt_corr = mk_safe(roccor.kSpreadMC, mu.charge, mu.pt, mu.eta, mu.phi, genMu.pt)
                if self.corr in ('up', 'down'):
                    pt_err = mk_safe(roccor.kSpreadMCerror, mu.charge, mu.pt, mu.eta, mu.phi, genMu.pt)
            else:
                self.rnd.SetSeed(rndSeed(event, mu))
                u1 = self.rnd.Uniform(0.0, 1.0)
                pt_corr = mk_safe(roccor.kSmearMC, mu.charge, mu.pt, mu.eta, mu.phi, mu.nTrackerLayers, u1)
                if self.corr in ('up', 'down'):
                    pt_err = mk_safe(roccor.kSmearMCerror, mu.charge, mu.pt, mu.eta, mu.phi, mu.nTrackerLayers, u1)
        else:
            pt_corr = mk_safe(roccor.kScaleDT, mu.charge, mu.pt, mu.eta, mu.phi)
            if self.corr in ('up', 'down'):
                pt_err = mk_safe(roccor.kScaleDTerror, mu.charge, mu.pt, mu.eta, mu.phi)

        if self.corr == 'up':
            pt_corr += pt_err
        elif self.corr == 'down':
            pt_corr -= pt_err
        mu._roccor = pt_corr
        mu.pt = max(0, mu.pt * pt_corr)

        if self.geofit:
            if abs(mu.dxybs) < 0.01:
                mu.pt = max(0, self._geofit(mu.dxybs * mu.charge, mu.pt, mu.eta, self.year))
