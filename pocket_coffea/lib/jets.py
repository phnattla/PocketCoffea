import copy
import importlib
import gzip
import cloudpickle

import awkward as ak
import numpy as np
import correctionlib
from coffea.jetmet_tools import  CorrectedMETFactory, JECStack, CorrectedJetsFactory
# from coffea.jetmet_tools import  JECStack
from ..lib.deltaR_matching import get_matching_pairs_indices, object_matching


def add_jec_variables(jets, event_rho, isMC=True):
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    if isMC:
        jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    print(jets.pt_raw)
    print(jets.rawFactor)
    return jets

def load_jet_factory(params):
    #read the factory file from params and load it
    with gzip.open(params.jets_calibration.factory_file) as fin:
        try:
            return cloudpickle.load(fin)
        except Exception as e:
            print(f"Error loading the jet factory file: {params.jets_calibration.factory_file} --> Please remove the file and rerun the code")
            raise Exception(f"Error loading the jet factory file: {params.jets_calibration.factory_file} --> Please remove the file and rerun the code")
        
jerc_dict = {
    "2016": {
        "jec_mc"  : "Summer19UL16_V7_MC",
        "jec_data": "Summer19UL16_RunFGH_V7_DATA",
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
        ],
        "jer"     : "Summer20UL16_JRV3_MC",
        "junc"    : [
            'FlavorQCD', 'FlavorPureBottom', 'FlavorPureQuark', 'FlavorPureGluon', 'FlavorPureCharm',
            'Regrouped_BBEC1', 'Regrouped_Absolute', 'Regrouped_RelativeBal', 'RelativeSample'
        ]
    },
    "2016APV": {
        "jec_mc": "Summer19UL16APV_V7_MC",
        "jec_data": {
            "B": "Summer19UL16APV_RunBCD_V7_DATA",
            "C": "Summer19UL16APV_RunBCD_V7_DATA",
            "D": "Summer19UL16APV_RunBCD_V7_DATA",
            "E": "Summer19UL16APV_RunEF_V7_DATA",
            "F": "Summer19UL16APV_RunEF_V7_DATA",
        },
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
        ],
        "jer": "Summer20UL16APV_JRV3_MC",
        "junc"    : [
            'FlavorQCD', 'FlavorPureBottom', 'FlavorPureQuark', 'FlavorPureGluon', 'FlavorPureCharm',
            'Regrouped_BBEC1', 'Regrouped_Absolute', 'Regrouped_RelativeBal', 'RelativeSample'
        ]
    },
    "2017": {
        "jec_mc": "Summer19UL17_V5_MC",
        "jec_data": {
            "B": "Summer19UL17_RunB_V5_DATA",
            "C": "Summer19UL17_RunC_V5_DATA",
            "D": "Summer19UL17_RunD_V5_DATA",
            "E": "Summer19UL17_RunE_V5_DATA",
            "F": "Summer19UL17_RunF_V5_DATA",
        },
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
        ],
        "jer": "Summer19UL17_JRV2_MC",
        "junc"    : [
            'FlavorQCD', 'FlavorPureBottom', 'FlavorPureQuark', 'FlavorPureGluon', 'FlavorPureCharm',
            'Regrouped_BBEC1', 'Regrouped_Absolute', 'Regrouped_RelativeBal', 'RelativeSample'
        ]
    },
    "2018": {
        "jec_mc": "Summer19UL18_V5_MC",
        "jec_data": {
            "A": "Summer19UL18_RunA_V5_DATA",
            "B": "Summer19UL18_RunB_V5_DATA",
            "C": "Summer19UL18_RunC_V5_DATA",
            "D": "Summer19UL18_RunD_V5_DATA",
        },
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
        ],
        "jer": "Summer19UL18_JRV2_MC",
        "junc"    : [
            'FlavorQCD', 'FlavorPureBottom', 'FlavorPureQuark', 'FlavorPureGluon', 'FlavorPureCharm',
            'Regrouped_BBEC1', 'Regrouped_Absolute', 'Regrouped_RelativeBal', 'RelativeSample'
        ]

    },
    "2022": {
        "jec_mc"  : "Summer22_22Sep2023_V2_MC",
        "jec_data": "Summer22_22Sep2023_RunCD_V2_DATA",
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
            "L3Absolute",
            "L2L3Residual",
        ],
        "jer"     : "Summer22_22Sep2023_JRV1_MC",
        "junc"    : [
            "AbsoluteMPFBias","AbsoluteScale","FlavorQCD","Fragmentation","PileUpDataMC",
            "PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef",
            "RelativeFSR","RelativeJERHF","RelativePtBB","RelativePtHF","RelativeBal",
            "SinglePionECAL","SinglePionHCAL",
            "AbsoluteStat","RelativeJEREC1","RelativeJEREC2","RelativePtEC1","RelativePtEC2",
            "TimePtEta","RelativeSample","RelativeStatEC","RelativeStatFSR","RelativeStatHF",
            "Total",
        ]
    },
    "2022EE": {
        "jec_mc": "Summer22EE_22Sep2023_V2_MC",
        "jec_data": {
            "E": "Summer22EE_22Sep2023_RunE_V2_DATA",
            "F": "Summer22EE_22Sep2023_RunF_V2_DATA",
            "G": "Summer22EE_22Sep2023_RunG_V2_DATA",
        },
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
            "L3Absolute",
            "L2L3Residual",
        ],
        "jer": "Summer22EE_22Sep2023_JRV1_MC",
        "junc"    : [
            "AbsoluteMPFBias","AbsoluteScale","FlavorQCD","Fragmentation","PileUpDataMC",
            "PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef",
            "RelativeFSR","RelativeJERHF","RelativePtBB","RelativePtHF","RelativeBal",
            "SinglePionECAL","SinglePionHCAL",
            "AbsoluteStat","RelativeJEREC1","RelativeJEREC2","RelativePtEC1","RelativePtEC2",
            "TimePtEta","RelativeSample","RelativeStatEC","RelativeStatFSR","RelativeStatHF",
            "Total",
        ]
    },
    "2023": {
        "jec_mc": "Summer23Prompt23_V1_MC",
        "jec_data": {
            "Cv1": "Summer23Prompt23_RunCv123_V1_DATA",
            "Cv2": "Summer23Prompt23_RunCv123_V1_DATA",
            "Cv3": "Summer23Prompt23_RunCv123_V1_DATA",
            "Cv4": "Summer23Prompt23_RunCv4_V1_DATA",
        },
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
            "L3Absolute",
            "L2L3Residual",
        ],
        "jer": "Summer23Prompt23_RunCv1234_JRV1_MC",
        "junc"    : [
            "AbsoluteMPFBias","AbsoluteScale","FlavorQCD","Fragmentation","PileUpDataMC",
            "PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef",
            "RelativeFSR","RelativeJERHF","RelativePtBB","RelativePtHF","RelativeBal",
            "SinglePionECAL","SinglePionHCAL",
            "AbsoluteStat","RelativeJEREC1","RelativeJEREC2","RelativePtEC1","RelativePtEC2",
            "TimePtEta","RelativeSample","RelativeStatEC","RelativeStatFSR","RelativeStatHF",
            "Total",
        ]
    },
    "2023BPix": {
        "jec_mc"  : "Summer23BPixPrompt23_V1_MC",
        "jec_data": "Summer23BPixPrompt23_RunD_V1_DATA",
        "jec_levels": [
            "L1FastJet",
            "L2Relative",
            "L3Absolute",
            "L2L3Residual",
        ],
        "jer"     : "Summer23BPixPrompt23_RunD_JRV1_MC",
        "junc"    : [
            "AbsoluteMPFBias","AbsoluteScale","FlavorQCD","Fragmentation","PileUpDataMC",
            "PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF","PileUpPtRef",
            "RelativeFSR","RelativeJERHF","RelativePtBB","RelativePtHF","RelativeBal",
            "SinglePionECAL","SinglePionHCAL",
            "AbsoluteStat","RelativeJEREC1","RelativeJEREC2","RelativePtEC1","RelativePtEC2",
            "TimePtEta","RelativeSample","RelativeStatEC","RelativeStatFSR","RelativeStatHF",
            "Total",
        ]
    }
}       

def get_jerc_keys(year, isdata, era=None):
    # Jet Algorithm
    if year.startswith("202"):
        jet_algo = 'AK4PFPuppi'
    else:
        jet_algo = 'AK4PFchs'

    #jec levels
    jec_levels = jerc_dict[year]['jec_levels']

    # jerc keys and junc types
    if not isdata:
        jec_key    = jerc_dict[year]['jec_mc']
        jer_key    = jerc_dict[year]['jer']
        junc_types = jerc_dict[year]['junc']
    else:
        if year in ['2016','2022','2023BPix']:
            jec_key = jerc_dict[year]['jec_data']
        else:
            jec_key = jerc_dict[year]['jec_data'][era]
        jer_key     = None
        junc_types  = None

    return jet_algo, jec_key, jec_levels, jer_key, junc_types



def get_jet_factory_corrlib(chunk_metadata):
    print("TEST JEC Corrlib")
    # year = chunk_metadata["year"]
    isData = not chunk_metadata["isMC"]
    print(isData)
    # print(params.keys())
    json_path = f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2023_Summer23/jet_jerc.json.gz"
    # json_path = params["JECjsonFiles"][year]["AK4"]
    print(json_path)
    jet_algo, jec_tag, jec_levels, jer_tag, junc_types = get_jerc_keys("2023", isData, chunk_metadata["era"])
    # json_path = topcoffea_path(f"data/POG/JME/{jec_year}/jet_jerc.json.gz")

    # Create JECStack for clib scenario
    jec_stack = JECStack(
        jec_tag=jec_tag,
        jec_levels=jec_levels,
        jer_tag=jer_tag,
        jet_algo=jet_algo,
        junc_types=junc_types,
        json_path=json_path,
        use_clib=True,
        savecorr=False
    )

    # Name map for jet or MET corrections
    name_map = {
        'JetPt': 'pt',
        'JetMass': 'mass',
        'JetEta': 'eta',
        'JetPhi': 'phi',
        'JetA': 'area',
        'ptGenJet': 'pt_gen',
        'ptRaw': 'pt_raw',
        'massRaw': 'mass_raw',
        'Rho': 'event_rho',
    }
    return CorrectedJetsFactory(name_map, jec_stack)


def jet_correction_corrlib(params, events, jets, factory, jet_type, chunk_metadata, cache):
    # print(type(factory["Data"][jet_type][chunk_metadata["year"]][chunk_metadata["era"]]))
    if chunk_metadata["year"] in ['2016_PreVFP', '2016_PostVFP','2017','2018']:
        rho = events.fixedGridRhoFastjetAll
    else:
        rho = events.Rho.fixedGridRhoFastjetAll

    if chunk_metadata["isMC"]:
        return get_jet_factory_corrlib(chunk_metadata).build(
            add_jec_variables(jets, rho, isMC=True), cache
        )
    else:
        # if chunk_metadata["era"] not in factory["Data"][jet_type][chunk_metadata["year"]]:
            # raise Exception(f"Factory for {jet_type} in {chunk_metadata['year']} and era {chunk_metadata['era']} not found. Check your jet calibration files.")

        return get_jet_factory_corrlib(chunk_metadata).build(
            add_jec_variables(jets, rho, isMC=False), cache
        )

def jet_correction(params, events, jets, factory, jet_type, chunk_metadata, cache):
    print(type(factory["Data"][jet_type][chunk_metadata["year"]][chunk_metadata["era"]]))
    if chunk_metadata["year"] in ['2016_PreVFP', '2016_PostVFP','2017','2018']:
        rho = events.fixedGridRhoFastjetAll
    else:
        rho = events.Rho.fixedGridRhoFastjetAll

    if chunk_metadata["isMC"]:
        return factory["MC"][jet_type][chunk_metadata["year"]].build(
            add_jec_variables(jets, rho, isMC=True), cache
        )
    else:
        if chunk_metadata["era"] not in factory["Data"][jet_type][chunk_metadata["year"]]:
            raise Exception(f"Factory for {jet_type} in {chunk_metadata['year']} and era {chunk_metadata['era']} not found. Check your jet calibration files.")

        return factory["Data"][jet_type][chunk_metadata["year"]][chunk_metadata["era"]].build(
            add_jec_variables(jets, rho, isMC=False), cache
        )

def met_correction_after_jec(events, METcoll, jets_pre_jec, jets_post_jec):
    '''This function rescale the MET vector by minus delta of the jets after JEC correction
    and before the jEC correction.
    This can be used also to rescale the MET when updating on the fly the JEC calibration. '''
    orig_tot_px = ak.sum(jets_pre_jec.px, axis=1)
    orig_tot_py = ak.sum(jets_pre_jec.py, axis=1)
    new_tot_px = ak.sum(jets_post_jec.px, axis=1)
    new_tot_py = ak.sum(jets_post_jec.py, axis=1)
    newpx =  events[METcoll].px - (new_tot_px - orig_tot_px) 
    newpy =  events[METcoll].py - (new_tot_py - orig_tot_py) 
    
    newMetPhi = np.arctan2(newpy, newpx)
    newMetPt = (newpx**2 + newpy**2)**0.5
    
    return  {"pt": newMetPt, "phi": newMetPhi}


def met_correction(params, MET, jets):
    met_factory = CorrectedMETFactory(params.jet_calibration.jec_name_map) # to be fixed
    return met_factory.build(MET, jets, {})
    
def met_xy_correction(params, events, METcol,  year, era):
    '''Apply MET xy corrections to MET collection'''
    metx = events[METcol].pt * np.cos(events[METcol].phi)
    mety = events[METcol].pt * np.sin(events[METcol].phi)
    nPV = events.PV.npvs

    if era == "MC":
        params_ = params["MET_xy"]["MC"][year]
    else:
        params_ = params["MET_xy"]["Data"][year][era]

    metx = metx - (params_[0][0] * nPV + params_[0][1])
    mety = mety - (params_[1][0] * nPV + params_[1][1])
    pt_corr = np.hypot(metx, mety)
    phi_corr = np.arctan2(mety, metx)
    
    return pt_corr, phi_corr


def jet_correction_correctionlib(
    events, Jet, typeJet, year, JECversion, JERversion=None, verbose=False
):
    '''
    This function implements the Jet Energy corrections and Jet energy smearning
    using factors from correctionlib common-POG json file
    example here: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/blob/master/examples/jercExample.py

    '''
    jsonfile = JECjsonFiles[year][
        [t for t in ['AK4', 'AK8'] if typeJet.startswith(t)][0]
    ]
    JECfile = correctionlib.CorrectionSet.from_file(jsonfile)
    corr = JECfile.compound[f'{JECversion}_L1L2L3Res_{typeJet}']

    # until correctionlib handles jagged data natively we have to flatten and unflatten
    jets = events[Jet]
    jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
    jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
    jets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
    j, nj = ak.flatten(jets), ak.num(jets)
    flatCorrFactor = corr.evaluate(
        np.array(j['area']),
        np.array(j['eta']),
        np.array(j['pt_raw']),
        np.array(j['rho']),
    )
    corrFactor = ak.unflatten(flatCorrFactor, nj)

    jets_corrected = copy.copy(jets)
    jets_corrected['pt'] = jets['pt_raw'] * corrFactor
    jets_corrected['mass'] = jets['mass_raw'] * corrFactor
    jets_corrected['rho'] = jets['rho']

    seed = events.event[0]

    if verbose:
        print()
        print(seed, 'JEC: starting columns:', ak.fields(jets), end='\n\n')

        print(seed, 'JEC: untransformed pt ratios', jets.pt / jets.pt_raw)
        print(seed, 'JEC: untransformed mass ratios', jets.mass / jets.mass_raw)

        print(
            seed, 'JEC: corrected pt ratios', jets_corrected.pt / jets_corrected.pt_raw
        )
        print(
            seed,
            'JEC: corrected mass ratios',
            jets_corrected.mass / jets_corrected.mass_raw,
        )

        print()
        print(seed, 'JEC: corrected columns:', ak.fields(jets_corrected), end='\n\n')

        # print('JES UP pt ratio',jets_corrected.JES_jes.up.pt/jets_corrected.pt_raw)
        # print('JES DOWN pt ratio',jets_corrected.JES_jes.down.pt/jets_corrected.pt_raw, end='\n\n')

    # Apply JER pt smearing (https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetResolution)
    # The hybrid scaling method is implemented: if a jet is matched to a gen-jet, the scaling method is applied;
    # if a jet is not gen-matched, the stochastic smearing is applied.
    if JERversion:
        sf = JECfile[f'{JERversion}_ScaleFactor_{typeJet}']
        res = JECfile[f'{JERversion}_PtResolution_{typeJet}']
        j, nj = ak.flatten(jets_corrected), ak.num(jets_corrected)
        scaleFactor_flat = sf.evaluate(j['eta'].to_numpy(), 'nom')
        ptResolution_flat = res.evaluate(
            j['eta'].to_numpy(), j['pt'].to_numpy(), j['rho'].to_numpy()
        )
        scaleFactor = ak.unflatten(scaleFactor_flat, nj)
        ptResolution = ak.unflatten(ptResolution_flat, nj)
        # Match jets with gen-level jets, with DeltaR and DeltaPt requirements
        dr_min = {'AK4PFchs': 0.2, 'AK8PFPuppi': 0.4}[
            typeJet
        ]  # Match jets within a cone with half the jet radius
        pt_min = (
            3 * ptResolution * jets_corrected['pt']
        )  # Match jets whose pt does not differ more than 3 sigmas from the gen-level pt
        genJet = {'AK4PFchs': 'GenJet', 'AK8PFPuppi': 'GenJetAK8'}[typeJet]
        genJetIdx = {'AK4PFchs': 'genJetIdx', 'AK8PFPuppi': 'genJetAK8Idx'}[typeJet]

        # They can be matched manually
        # matched_genjets, matched_jets, deltaR_matched = object_matching(genjets, jets_corrected, dr_min, pt_min)
        # Or the association in NanoAOD it can be used, removing the indices that are not found. That happens because
        # not all the genJet are saved in the NanoAODs.
        genjets = events[genJet]
        Ngenjet = ak.num(genjets)
        matched_genjets_idx = ak.mask(
            jets_corrected[genJetIdx],
            (jets_corrected[genJetIdx] < Ngenjet) & (jets_corrected[genJetIdx] != -1),
        )
        # this array of indices has already the dimension of the Jet collection
        # in NanoAOD nomatch == -1 --> convert to None with a mask
        matched_objs_mask = ~ak.is_none(matched_genjets_idx, axis=1)
        matched_genjets = genjets[matched_genjets_idx]
        matched_jets = ak.mask(jets_corrected, matched_objs_mask)

        deltaPt = ak.unflatten(
            np.abs(ak.flatten(matched_jets.pt) - ak.flatten(matched_genjets.pt)),
            ak.num(matched_genjets),
        )
        matched_genjets = ak.mask(matched_genjets, deltaPt < pt_min)
        matched_jets = ak.mask(matched_jets, deltaPt < pt_min)

        # Compute energy correction factor with the scaling method
        detSmear = (
            1
            + (scaleFactor - 1)
            * (matched_jets['pt'] - matched_genjets['pt'])
            / matched_jets['pt']
        )
        # Compute energy correction factor with the stochastic method
        np.random.seed(seed)
        seed_dict = {}
        filename = events.metadata['filename']
        entrystart = events.metadata['entrystart']
        entrystop = events.metadata['entrystop']
        seed_dict[f'chunk_{filename}_{entrystart}-{entrystop}'] = seed
        rand_gaus = np.random.normal(
            np.zeros_like(ptResolution_flat), ptResolution_flat
        )
        jersmear = ak.unflatten(rand_gaus, nj)
        sqrt_arg_flat = scaleFactor_flat**2 - 1
        sqrt_arg_flat = ak.where(
            sqrt_arg_flat > 0, sqrt_arg_flat, ak.zeros_like(sqrt_arg_flat)
        )
        sqrt_arg = ak.unflatten(sqrt_arg_flat, nj)
        stochSmear = 1 + jersmear * np.sqrt(sqrt_arg)
        isMatched = ~ak.is_none(matched_jets.pt, axis=1)
        smearFactor = ak.where(isMatched, detSmear, stochSmear)

        jets_smeared = copy.copy(jets_corrected)
        jets_smeared['pt'] = jets_corrected['pt'] * smearFactor
        jets_smeared['mass'] = jets_corrected['mass'] * smearFactor

        if verbose:
            print()
            print(seed, "JER: isMatched", isMatched)
            print(seed, "JER: matched_jets.pt", matched_jets.pt)
            print(seed, "JER: smearFactor", smearFactor, end='\n\n')

            print(
                seed,
                'JER: corrected pt ratios',
                jets_corrected.pt / jets_corrected.pt_raw,
            )
            print(
                seed,
                'JER: corrected mass ratios',
                jets_corrected.mass / jets_corrected.mass_raw,
            )

            print(seed, 'JER: smeared pt ratios', jets_smeared.pt / jets_corrected.pt)
            print(
                seed,
                'JER: smeared mass ratios',
                jets_smeared.mass / jets_corrected.mass,
            )

            print()
            print(seed, 'JER: corrected columns:', ak.fields(jets_smeared), end='\n\n')

        return jets_smeared, seed_dict
    else:
        return jets_corrected


def jet_selection(events, jet_type, params, year, leptons_collection="", jet_tagger=""):

    jets = events[jet_type]
    cuts = params.object_preselection[jet_type]
    # Only jets that are more distant than dr to ALL leptons are tagged as good jets
    # Mask for  jets not passing the preselection
    mask_presel = (
        (jets.pt > cuts["pt"])
        & (np.abs(jets.eta) < cuts["eta"])
        & (jets.jetId >= cuts["jetId"])
    )
    # Lepton cleaning
    if leptons_collection != "":
        dR_jets_lep = jets.metric_table(events[leptons_collection])
        mask_lepton_cleaning = ak.prod(dR_jets_lep > cuts["dr_lepton"], axis=2) == 1
    else:
        mask_lepton_cleaning = True

    if jet_type == "Jet":
        # Selection on PUid. Only available in Run2 UL, thus we need to determine which sample we run over;
        if year in ['2016_PreVFP', '2016_PostVFP','2017','2018']:
            mask_jetpuid = (jets.puId >= params.jet_scale_factors.jet_puId[year]["working_point"][cuts["puId"]["wp"]]) | (
                jets.pt >= cuts["puId"]["maxpt"]
            )
        else:
            mask_jetpuid = True
  
        mask_good_jets = mask_presel & mask_lepton_cleaning & mask_jetpuid

        if jet_tagger != "":
            if "PNet" in jet_tagger:
                B   = "btagPNetB"
                CvL = "btagPNetCvL"
                CvB = "btagPNetCvB"
            elif "DeepFlav" in jet_tagger:
                B   = "btagDeepFlavB"
                CvL = "btagDeepFlavCvL"
                CvB = "btagDeepFlavCvB"
            elif "RobustParT" in jet_tagger:
                B   = "btagRobustParTAK4B"
                CvL = "btagRobustParTAK4CvL"
                CvB = "btagRobustParTAK4CvB"
            else:
                raise NotImplementedError(f"This tagger is not implemented: {jet_tagger}")
            
            if B not in jets.fields or CvL not in jets.fields or CvB not in jets.fields:
                raise NotImplementedError(f"{B}, {CvL}, and/or {CvB} are not available in the input.")

            jets["btagB"] = jets[B]
            jets["btagCvL"] = jets[CvL]
            jets["btagCvB"] = jets[CvB]

    elif jet_type == "FatJet":
        # Apply the msd and preselection cuts
        mask_msd = events.FatJet.msoftdrop > cuts["msd"]
        mask_good_jets = mask_presel & mask_msd

        if jet_tagger != "":
            if "PNetMD" in jet_tagger:
                BB   = "particleNet_XbbVsQCD"
                CC   = "particleNet_XccVsQCD"
            elif "PNet" in jet_tagger:
                BB   = "particleNetWithMass_HbbvsQCD"
                CC   = "particleNetWithMass_HccvsQCD"
            else:
                raise NotImplementedError(f"This tagger is not implemented: {jet_tagger}")
            
            if BB not in jets.fields or CC not in jets.fields:
                raise NotImplementedError(f"{BB} and/or {CC} are not available in the input.")

            jets["btagBB"] = jets[BB]
            jets["btagCC"] = jets[CC]

    return jets[mask_good_jets], mask_good_jets


def btagging(Jet, btag, wp, veto=False):
    if veto:
        return Jet[Jet[btag["btagging_algorithm"]] < btag["btagging_WP"][wp]]
    else:
        return Jet[Jet[btag["btagging_algorithm"]] > btag["btagging_WP"][wp]]


def CvsLsorted(jets,temp=None):    
    if temp is not None:
        raise NotImplementedError(f"Using the tagger name while calling `CvsLsorted` is deprecated. Please use `jet_tagger={temp}` as an argument to `jet_selection`.")
    return jets[ak.argsort(jets["btagCvL"], axis=1, ascending=False)]

def ProbBsorted(jets,temp=None):    
    if temp is not None:
        raise NotImplementedError(f"Using the tagger name while calling `ProbBsorted` is deprecated. Please use `jet_tagger={temp}` as an argument to `jet_selection`.")
    return jets[ak.argsort(jets["btagB"], axis=1, ascending=False)]


def get_dijet(jets, taggerVars=True, remnant_jet = False):
    if isinstance(taggerVars,str):
        raise NotImplementedError(f"Using the tagger name while calling `get_dijet` is deprecated. Please use `jet_tagger={taggerVars}` as an argument to `jet_selection`.")
    
    fields = {
        "pt": 0.,
        "eta": 0.,
        "phi": 0.,
        "mass": 0.,
    }
    
    if remnant_jet:
        fields_remnant = {
        "pt": 0.,
        "eta": 0.,
        "phi": 0.,
        "mass": 0.,
    }

    jets = ak.pad_none(jets, 2)
    njet = ak.num(jets[~ak.is_none(jets, axis=1)])
    
    dijet = jets[:, 0] + jets[:, 1]
    if remnant_jet:
        remnant = jets[:, 2:]

    for var in fields.keys():
        fields[var] = ak.where(
            (njet >= 2),
            getattr(dijet, var),
            fields[var]
        )
        
    if remnant_jet:
        for var in fields_remnant.keys():
            fields_remnant[var] = ak.where(
                (njet > 2),
                ak.sum(getattr(remnant, var), axis=1),
                fields_remnant[var]
            )

    fields["deltaR"] = ak.where( (njet >= 2), jets[:,0].delta_r(jets[:,1]), -1)
    fields["deltaPhi"] = ak.where( (njet >= 2), abs(jets[:,0].delta_phi(jets[:,1])), -1)
    fields["deltaEta"] = ak.where( (njet >= 2), abs(jets[:,0].eta - jets[:,1].eta), -1)
    fields["j1Phi"] = ak.where( (njet >= 2), jets[:,0].phi, -1)
    fields["j2Phi"] = ak.where( (njet >= 2), jets[:,1].phi, -1)
    fields["j1pt"] = ak.where( (njet >= 2), jets[:,0].pt, -1)
    fields["j2pt"] = ak.where( (njet >= 2), jets[:,1].pt, -1)
    fields["j1eta"] = ak.where( (njet >= 2), jets[:,0].eta, -1)
    fields["j2eta"] = ak.where( (njet >= 2), jets[:,1].eta, -1)
    fields["j1mass"] = ak.where( (njet >= 2), jets[:,0].mass, -1)
    fields["j2mass"] = ak.where( (njet >= 2), jets[:,1].mass, -1)


    if "jetId" in jets.fields and taggerVars:
        '''This dijet fuction should work for GenJets as well. But the btags are not available for them
        Thus, one has to check if a Jet is a GenJet or reco Jet. The jetId variable is only available in reco Jets'''
        fields["j1CvsL"] = ak.where( (njet >= 2), jets[:,0]["btagCvL"], -1)
        fields["j2CvsL"] = ak.where( (njet >= 2), jets[:,1]["btagCvL"], -1)
        fields["j1CvsB"] = ak.where( (njet >= 2), jets[:,0]["btagCvB"], -1)
        fields["j2CvsB"] = ak.where( (njet >= 2), jets[:,1]["btagCvB"], -1)
    
    dijet = ak.zip(fields, with_name="PtEtaPhiMCandidate")
    if remnant_jet:
        remnant = ak.zip(fields_remnant, with_name="PtEtaPhiMCandidate")

    if not remnant_jet:
        return dijet
    else:
        return dijet, remnant
