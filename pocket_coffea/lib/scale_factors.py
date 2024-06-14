import numpy as np
import awkward as ak
import correctionlib


def get_ele_sf(
    params, year, pt, eta, counts=None, key='', pt_region=None, variations=["nominal"]
):
    '''
    This function computes the per-electron reco or id SF.
    If 'reco', the appropriate corrections are chosen by using the argument `pt_region`.
    '''
    electronSF = params["lepton_scale_factors"]["electron_sf"]

    if key in ['reco', 'id']:
        electron_correctionset = correctionlib.CorrectionSet.from_file(
            electronSF.JSONfiles[year]["file"]
        )
        map_name = electronSF.JSONfiles[year]["name"]

        if key == 'reco':
            sfname = electronSF.JSONfiles[year]["reco"][pt_region]
        elif key == 'id':
            sfname = electronSF["id"][params.object_preselection["Electron"]["id"]]

        # translate the `year` key into the corresponding key in the correction file provided by the EGM-POG
        year_pog = electronSF["era_mapping"][year]

        sf = electron_correctionset[map_name].evaluate(
            year_pog, "sf", sfname, eta.to_numpy(), pt.to_numpy()
        )
        sfup = electron_correctionset[map_name].evaluate(
            year_pog, "sfup", sfname, eta.to_numpy(), pt.to_numpy()
        )
        sfdown = electron_correctionset[map_name].evaluate(
            year_pog, "sfdown", sfname, eta.to_numpy(), pt.to_numpy()
        )
        # The unflattened arrays are returned in order to have one row per event.
        return (
            ak.unflatten(sf, counts),
            ak.unflatten(sfup, counts),
            ak.unflatten(sfdown, counts),
        )
    elif key == 'trigger':

        electron_correctionset = correctionlib.CorrectionSet.from_file(
            electronSF.trigger_sf[year]["file"]
        )
        map_name = electronSF.trigger_sf[year]["name"]

        output = {}
        for variation in variations:
            if variation == "nominal":
                output[variation] = [
                    electron_correctionset[map_name].evaluate(
                        variation, pt.to_numpy(), eta.to_numpy()
                    )
                ]
            else:
                # Nominal sf==1
                nominal = np.ones_like(pt.to_numpy())
                # Systematic variations
                output[variation] = [
                    nominal,
                    electron_correctionset[map_name].evaluate(
                        f"{variation}Up", pt.to_numpy(), eta.to_numpy()
                    ),
                    electron_correctionset[map_name].evaluate(
                        f"{variation}Down", pt.to_numpy(), eta.to_numpy()
                    ),
                ]
            for i, sf in enumerate(output[variation]):
                output[variation][i] = ak.unflatten(sf, counts)

        return output


def get_mu_sf(params, year, pt, eta, counts, key=''):
    '''
    This function computes the per-muon id or iso SF.
    '''
    muonSF = params["lepton_scale_factors"]["muon_sf"]

    muon_correctionset = correctionlib.CorrectionSet.from_file(
        muonSF.JSONfiles[year]['file']
    )
    
    sfName = muonSF.sf_name[year][key]
    
    sf = muon_correctionset[sfName].evaluate(
        np.abs(eta.to_numpy()), pt.to_numpy(), "nominal"
    )
    sfup = muon_correctionset[sfName].evaluate(
        np.abs(eta.to_numpy()), pt.to_numpy(), "systup"
    )
    sfdown = muon_correctionset[sfName].evaluate(
        np.abs(eta.to_numpy()), pt.to_numpy(), "systdown"
    )
    
    # The unflattened arrays are returned in order to have one row per event.
    return (
        ak.unflatten(sf, counts),
        ak.unflatten(sfup, counts),
        ak.unflatten(sfdown, counts),
    )


def sf_ele_reco(params, events, year):
    '''
    This function computes the per-electron reco SF and returns the corresponding per-event SF, obtained by multiplying the per-electron SF in each event.
    Additionally, also the up and down variations of the SF are returned.
    Electrons are split into two categories based on a pt cut depending on the Run preiod, so that the proper SF is applied.
    '''
    ele_pt = events.ElectronGood.pt
    ele_eta = events.ElectronGood.etaSC

    pt_ranges = []
    if year in ['2016_PreVFP', '2016_PostVFP','2017','2018']:
        pt_ranges += [("pt_lt_20", (ele_pt < 20)), 
                      ("pt_gt_20", (ele_pt >= 20))]
    elif year in ["2022_preEE", "2022_postEE"]:
        pt_ranges += [("pt_lt_20", (ele_pt < 20)), 
                      ("pt_gt_20_lt_75", (ele_pt >= 20) & (ele_pt <75)), 
                      ("pt_gt_75", (ele_pt >= 75))]
    else:
        raise Exception("For chosen year "+year+" sf_ele_reco are not implemented yet")
    
    sf_reco, sfup_reco, sfdown_reco = [], [], []

    for pt_range_key, pt_range in pt_ranges:
        ele_pt_inPtRange = ak.flatten(ele_pt[pt_range])
        ele_eta_inPtRange = ak.flatten(ele_eta[pt_range])
        ele_counts_inPtRange = ak.num(ele_pt[pt_range])

        sf_reco_inPtRange, sfup_reco_inPtRange, sfdown_reco_inPtRange = get_ele_sf(
            params,
            year,
            ele_pt_inPtRange,
            ele_eta_inPtRange,
            ele_counts_inPtRange,
            'reco',
            pt_range_key,
        )
        sf_reco.append(sf_reco_inPtRange)
        sfup_reco.append(sfup_reco_inPtRange)
        sfdown_reco.append(sfdown_reco_inPtRange)

    sf_reco = ak.prod(
        ak.concatenate(sf_reco, axis=1), axis=1
    )
    sfup_reco = ak.prod(
        ak.concatenate(sfup_reco, axis=1), axis=1
    )
    sfdown_reco = ak.prod(
        ak.concatenate(sfdown_reco, axis=1), axis=1
    )

    return sf_reco, sfup_reco, sfdown_reco


def sf_ele_id(params, events, year):
    '''
    This function computes the per-electron id SF and returns the corresponding per-event SF, obtained by multiplying the per-electron SF in each event.
    Additionally, also the up and down variations of the SF are returned.
    '''
    ele_pt = events.ElectronGood.pt
    ele_eta = events.ElectronGood.etaSC

    ele_pt_flat, ele_eta_flat, ele_counts = (
        ak.flatten(ele_pt),
        ak.flatten(ele_eta),
        ak.num(ele_pt),
    )
    sf_id, sfup_id, sfdown_id = get_ele_sf(
        params, year, ele_pt_flat, ele_eta_flat, ele_counts, 'id'
    )

    # The SF arrays corresponding to the electrons are multiplied along the electron axis in order to obtain a per-event scale factor.
    return ak.prod(sf_id, axis=1), ak.prod(sfup_id, axis=1), ak.prod(sfdown_id, axis=1)


def sf_ele_trigger(params, events, year, variations=["nominal"]):
    '''
    This function computes the semileptonic electron trigger SF by considering the leading electron in the event.
    This computation is valid only in the case of the semileptonic final state.
    Additionally, also the up and down variations of the SF for a set of systematic uncertainties are returned.
    '''

    ele_pt = events.ElectronGood.pt
    ele_eta = events.ElectronGood.etaSC

    ele_pt_flat, ele_eta_flat, ele_counts = (
        ak.flatten(ele_pt),
        ak.flatten(ele_eta),
        ak.num(ele_pt),
    )
    sf_dict = get_ele_sf(
        params,
        year,
        ele_pt_flat,
        ele_eta_flat,
        ele_counts,
        'trigger',
        variations=variations,
    )

    for variation in sf_dict.keys():
        for i, sf in enumerate(sf_dict[variation]):
            sf_dict[variation][i] = ak.prod(sf, axis=1)

    return sf_dict


def sf_mu(params, events, year, key=''):
    '''
    This function computes the per-muon id SF and returns the corresponding per-event SF, obtained by multiplying the per-muon SF in each event.
    Additionally, also the up and down variations of the SF are returned.
    '''
    mu_pt = events.MuonGood.pt
    mu_eta = events.MuonGood.eta

    # Since `correctionlib` does not support jagged arrays as an input, the pt and eta arrays are flattened.
    mu_pt_flat, mu_eta_flat, mu_counts = (
        ak.flatten(mu_pt),
        ak.flatten(mu_eta),
        ak.num(mu_pt),
    )
    sf, sfup, sfdown = get_mu_sf(params, year, mu_pt_flat, mu_eta_flat, mu_counts, key)

    # The SF arrays corresponding to all the muons are multiplied along the
    # muon axis in order to obtain a per-event scale factor.
    return ak.prod(sf, axis=1), ak.prod(sfup, axis=1), ak.prod(sfdown, axis=1)

def sf_btag_fixed_wp(params, Jets, year, variations=["central"]):
    btag_algo = params["btagging"]["working_point"][year]["btagging_algorithm"]
    btag_wps = params["btagging"]["working_point"][year]["btagging_WP"]
    sf_file = params["jet_scale_factors"]["btagSF"][year]["file"]
    # Jets = events["JetGood"]

    btag_effi_corr_set = correctionlib.CorrectionSet.from_file(
        params.btagSF_calibration[year]["file"]
    )
    btag_sf_fixed_wp_corr_set = correctionlib.CorrectionSet.from_file(sf_file)

    jetpt = ak.flatten(Jets["pt"])
    jeteta = ak.flatten(Jets["eta"])
    jetflav = ak.flatten(Jets["hadronFlavour"])
    jetcounts = ak.num(Jets["pt"])

    wpname = {
        "L": "effiloose",
        "M": "effimedium",
        "T": "effitight"
    }
    
    jetpt_heavy = ak.flatten(Jets["pt"][Jets["hadronFlavour"]>3])
    jeteta_heavy = ak.flatten(Jets["eta"][Jets["hadronFlavour"]>3])
    jetflav_heavy = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"]>3])
    
    jetpt_light = ak.flatten(Jets["pt"][Jets["hadronFlavour"]<4])
    jeteta_light = ak.flatten(Jets["eta"][Jets["hadronFlavour"]<4])
    jetflav_light = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"]<4])

    wp = params.object_preselection["Jet"]["btag"]["wp"]

    effi_MC = ak.unflatten(
        btag_effi_corr_set[wpname[wp]].evaluate(jetpt, jeteta, jetflav), 
        counts=jetcounts
    )

    sf_light_flat = btag_sf_fixed_wp_corr_set["robustParticleTransformer_light"].evaluate(
        "central", 
        wp, 
        jetflav_light,
        abs(jeteta_light), 
        jetpt_light
    )
    sf_heavy_flat = btag_sf_fixed_wp_corr_set["robustParticleTransformer_comb"].evaluate(
        "central", 
        wp, 
        jetflav_heavy, 
        abs(jeteta_heavy), 
        jetpt_heavy
    )
    sf_flat = ak.to_numpy(ak.copy(jetpt))
    sf_flat[jetflav>3] = sf_heavy_flat
    sf_flat[jetflav<4] = sf_light_flat
    sf_DATA_MC = ak.unflatten(sf_flat, jetcounts)

    effi_DATA = ak.prod([sf_DATA_MC, effi_MC], axis=0)

    Jets = ak.with_field(Jets, effi_MC, "effi_MC_"+wp)
    Jets = ak.with_field(Jets, effi_DATA, "effi_DATA_"+wp)

    Jets_not_tagged = Jets[Jets[btag_algo] <= btag_wps[wp]]
    Jets_tagged = Jets[Jets[btag_algo] > btag_wps[wp]]

    p_MC_1 = 1 - Jets_not_tagged["effi_MC_"+wp]
    p_MC_2 = Jets_tagged["effi_MC_"+wp]

    p_DATA_1 = 1 - Jets_not_tagged["effi_DATA_"+wp]
    p_DATA_2 = Jets_tagged["effi_DATA_"+wp]

    p_MC = ak.concatenate([p_MC_1, p_MC_2], axis=1)
    p_MC = ak.prod(p_MC, axis=-1)
    p_DATA = ak.concatenate([p_DATA_1, p_DATA_2], axis=1)
    p_DATA = ak.prod(p_DATA, axis=-1)

    btag_sf_fixed_wp = np.divide(p_DATA, p_MC)
    # print(btag_sf_fixed_wp)
    return btag_sf_fixed_wp, ak.copy(btag_sf_fixed_wp), ak.copy(btag_sf_fixed_wp)

def sf_btag_fixed_wp_LMT(params, Jets, year, variations=["central"]):
    btag_algo = params["btagging"]["working_point"][year]["btagging_algorithm"]
    btag_wps = params["btagging"]["working_point"][year]["btagging_WP"]
    sf_file = params["jet_scale_factors"]["btagSF"][year]["file"]
    # Jets = events["JetGood"]

    btag_effi_corr_set = correctionlib.CorrectionSet.from_file(
        params.btagSF_calibration[year]["file"]
    )
    btag_sf_fixed_wp_corr_set = correctionlib.CorrectionSet.from_file(sf_file)

    jetpt = ak.flatten(Jets["pt"])
    jeteta = ak.flatten(Jets["eta"])
    jetflav = ak.flatten(Jets["hadronFlavour"])
    jetcounts = ak.num(Jets["pt"])

    wpname = {
        "L": "effiloose",
        "M": "effimedium",
        "T": "effitight"
    }
    
    jetpt_heavy = ak.flatten(Jets["pt"][Jets["hadronFlavour"]>3])
    jeteta_heavy = ak.flatten(Jets["eta"][Jets["hadronFlavour"]>3])
    jetflav_heavy = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"]>3])
    
    jetpt_light = ak.flatten(Jets["pt"][Jets["hadronFlavour"]<4])
    jeteta_light = ak.flatten(Jets["eta"][Jets["hadronFlavour"]<4])
    jetflav_light = ak.flatten(Jets["hadronFlavour"][Jets["hadronFlavour"]<4])

    for wp in ["L", "M", "T"]:
        effi_MC = ak.unflatten(
            btag_effi_corr_set[wpname[wp]].evaluate(jetpt, jeteta, jetflav), 
            counts=jetcounts
        )

        sf_light_flat = btag_sf_fixed_wp_corr_set["robustParticleTransformer_light"].evaluate(
            "central", 
            wp, 
            jetflav_light,
            abs(jeteta_light), 
            jetpt_light
        )
        sf_heavy_flat = btag_sf_fixed_wp_corr_set["robustParticleTransformer_comb"].evaluate(
            "central", 
            wp, 
            jetflav_heavy, 
            abs(jeteta_heavy), 
            jetpt_heavy
        )
        sf_flat = ak.to_numpy(ak.copy(jetpt))
        sf_flat[jetflav>3] = sf_heavy_flat
        sf_flat[jetflav<4] = sf_light_flat
        sf_DATA_MC = ak.unflatten(sf_flat, jetcounts)

        effi_DATA = ak.prod([sf_DATA_MC, effi_MC], axis=0)

        Jets = ak.with_field(Jets, effi_MC, "effi_MC_"+wp)
        Jets = ak.with_field(Jets, effi_DATA, "effi_DATA_"+wp)

    Jets_not_L = Jets[Jets[btag_algo] <= btag_wps["L"]]
    Jets_L_not_M = Jets[(Jets[btag_algo] > btag_wps["L"])
                            & (Jets[btag_algo] <= btag_wps["M"])]
    Jets_M_not_T = Jets[(Jets[btag_algo] > btag_wps["M"])
                            & (Jets[btag_algo] <= btag_wps["T"])]
    Jets_T = Jets[Jets[btag_algo] > btag_wps["T"]]

    p_MC_1 = 1 - Jets_not_L["effi_MC_L"]
    p_MC_2 = Jets_L_not_M["effi_MC_L"] - Jets_L_not_M["effi_MC_M"]
    p_MC_3 = Jets_M_not_T["effi_MC_M"] - Jets_M_not_T["effi_MC_T"]
    p_MC_4 = Jets_T["effi_MC_T"]

    p_DATA_1 = 1 - Jets_not_L["effi_DATA_L"]
    p_DATA_2 = Jets_L_not_M["effi_DATA_L"] - Jets_L_not_M["effi_DATA_M"]
    p_DATA_3 = Jets_M_not_T["effi_DATA_M"] - Jets_M_not_T["effi_DATA_T"]
    p_DATA_4 = Jets_T["effi_DATA_T"]

    p_MC = ak.concatenate([p_MC_1, p_MC_2, p_MC_3, p_MC_4], axis=1)
    p_DATA = ak.concatenate([p_DATA_1, p_DATA_2, p_DATA_3, p_DATA_4], axis=1)

    p_MC = ak.prod(p_MC, axis=-1)
    p_DATA = ak.prod(p_DATA, axis=-1)
    btag_sf_fixed_wp_LMT = np.divide(p_DATA, p_MC)

    return btag_sf_fixed_wp_LMT, ak.copy(btag_sf_fixed_wp_LMT), ak.copy(btag_sf_fixed_wp_LMT)

def sf_top_pt(part):
    part = part[~ak.is_none(part.parent, axis=1)]
    part = part[part.hasFlags("isLastCopy")]
    part = part[abs(part.pdgId) == 6]
    part = part[ak.argsort(part.pdgId, ascending=False)]
    arg = {
        "a": 0.103,
        "b": -0.0118, 
        "c": -0.000134,
        "d": 0.973
    }
    top_weight = arg["a"] * np.exp(arg["b"] * part.pt[:,0]) + arg["c"] * part.pt[:,0] + arg["d"]
    antitop_weight = arg["a"] * np.exp(arg["b"] * part.pt[:,1]) + arg["c"] * part.pt[:,1] + arg["d"]
    weight = np.sqrt(ak.prod([top_weight, antitop_weight], axis=0))
    return weight, ak.copy(weight), ak.copy(weight)

def sf_btag(params, jets, year, njets, variations=["central"]):
    '''
    DeepJet AK4 btagging SF.
    See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/BTV_2018_UL_btagging.html
    The scale factors have 8 default uncertainty
    sources (hf,lf,hfstats1/2,lfstats1/2,cferr1/2) (all of this up_*var*, and down_*var*).
    All except the cferr1/2 uncertainties are to be
    applied to light and b jets. The cferr1/2 uncertainties are to be applied to c jets.
    hf/lfstats1/2 uncertainties are to be decorrelated between years, the others correlated.
    Additional jes-varied scale factors are supplied to be applied for the jes variations.

    if variation is not one of the jes ones both the up and down sf is returned.
    If variation is a jet variation the argument must be up_jes* or down_jes* since it is applied on the specified
    Jes variation jets.
    '''
    btagSF = params.jet_scale_factors.btagSF[year]
    btag_discriminator = params.btagging.working_point[year]["btagging_algorithm"]
    cset = correctionlib.CorrectionSet.from_file(btagSF.file)
    corr = cset[btagSF.name]

    flavour = ak.to_numpy(ak.flatten(jets.hadronFlavour))
    abseta = np.abs(ak.to_numpy(ak.flatten(jets.eta)))
    pt = ak.to_numpy(ak.flatten(jets.pt))
    discr = ak.to_numpy(ak.flatten(jets[btag_discriminator]))

    central_SF_byjet = corr.evaluate("central", flavour, abseta, pt, discr)

    def _get_sf_variation_with_mask(variation, mask):
        index = (np.indices(discr.shape)).flatten()[mask]
        # Copying the central SF
        sf = np.copy(central_SF_byjet)
        w = corr.evaluate(variation, flavour[mask], abseta[mask], pt[mask], discr[mask])
        sf[index] = w
        sf_out = ak.prod(ak.unflatten(sf, njets), axis=1)
        return sf_out

    output = {}
    for variation in variations:
        if variation == "central":
            output[variation] = [ak.prod(ak.unflatten(central_SF_byjet, njets), axis=1)]
        else:
            # Nominal sf==1
            nominal = np.ones(ak.num(njets, axis=0))
            # Systematic variations
            if "cferr" in variation:
                # Computing the scale factor only on c-flavour jets
                c_mask = flavour == 4
                output[variation] = [
                    nominal,
                    _get_sf_variation_with_mask(f"up_{variation}", c_mask),
                    _get_sf_variation_with_mask(f"down_{variation}", c_mask),
                ]

            elif "JES" in variation:
                # We need to convert the name of the variation
                # from JES_VariationUp to  up_jesVariation
                if variation == "JES_TotalUp":
                    btag_jes_var = "up_jes"
                elif variation == "JES_TotalDown":
                    btag_jes_var = "down_jes"
                else:
                    if variation[-2:] == "Up":
                        btag_jes_var = f"up_jes{variation[4:-2]}"
                    elif variation[-4:] == "Down":
                        btag_jes_var = f"down_jes{variation[4:-4]}"

                # This is a special case where a dedicate btagSF is computed for up and down Jes shape variations.
                # This is not an up/down variation, but a single modified SF.
                # N.B: It is a central SF
                # notc_mask = flavour != 4
                # output["central"] = [_get_sf_variation_with_mask(btag_jes_var, notc_mask)]
                output["central"] = [
                    ak.prod(
                        ak.unflatten(
                            corr.evaluate("central", flavour, abseta, pt, discr), njets
                        ),
                        axis=1,
                    )
                ]
            else:
                # Computing the scale factor only NON c-flavour jets
                notc_mask = flavour != 4
                output[variation] = [
                    nominal,
                    _get_sf_variation_with_mask(f"up_{variation}", notc_mask),
                    _get_sf_variation_with_mask(f"down_{variation}", notc_mask),
                ]

    return output


def sf_btag_calib(params, sample, year, njets, jetsHt):
    '''Correction to btagSF computing by comparing the inclusive shape without btagSF and with btagSF in 2D:
    njets-JetsHT bins. Each sample/year has a different correction stored in the correctionlib format.'''
    cset = correctionlib.CorrectionSet.from_file(
        params.btagSF_calibration[year]["file"]
    )
    corr = cset[params.btagSF_calibration[year]["name"]]
    w = corr.evaluate(sample, year, ak.to_numpy(njets), ak.to_numpy(jetsHt))
    return w


def sf_ctag(params, jets, year, njets, variations=["central"]):
    '''
    Shape correction scale factors (SF) for DepJet charm tagger, taken from:
    https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/summaries/BTV_201X_UL_ctagging.html
    SFs are obtained per jet, then multiplied to obtain an overall weight per event.
    Note: this SF does not preserve the normalization of the MC samples!
     One has to re-normalize the MC samples using the sf_ctag_calib() method.
     The norm. calibration corrections are phase-space/analysis dependant.
     Therefore one has to derive them for each analysis separately.
    '''
    # print("Doing sf_ctag", year)

    ctagSF = params.jet_scale_factors.ctagSF[year]
    ctagger = params.ctagging.working_point[year]["tagger"]
    cset = correctionlib.CorrectionSet.from_file(ctagSF.SF_file)

    #print(list(cset.keys()))
    #print(list(cset.items()))

    corr = cset[ctagSF.name]

    #print(corr)

    flav = ak.to_numpy(ak.flatten(jets.hadronFlavour))
    CvL = ak.to_numpy(ak.flatten(jets.btagDeepCvL))
    CvB = ak.to_numpy(ak.flatten(jets.btagDeepCvB))

    central_SF_byjet = corr.evaluate("central", flav, CvL, CvB)

    #print(central_SF_byjet)
    #print("Unflatt=", ak.unflatten(central_SF_byjet, njets))
    #print("Prod:", ak.prod(ak.unflatten(central_SF_byjet, njets), axis=1))

    output = {}
    for variation in variations:
        if variation == "central":
            output[variation] = [ak.prod(ak.unflatten(central_SF_byjet, njets), axis=1)]
        else:
            # Nominal sf==1
            nominal = np.ones(ak.num(njets, axis=0))
            # Systematic variations
            up_variation_SF_byjet = corr.evaluate(f"up_{variation}", flav, CvL, CvB)
            down_variation_SF_byjet = corr.evaluate(f"down_{variation}", flav, CvL, CvB)

            output[variation] = [
                nominal,
                ak.prod(ak.unflatten(up_variation_SF_byjet,njets), axis=1),
                ak.prod(ak.unflatten(down_variation_SF_byjet,njets), axis=1)
            ]

    return output


def sf_ctag_calib(params, dataset, year, njets, jetsHt):
    '''
    These are correctiosn to normalization of every dataset after application of the ctag_sf shape correction.
    It was computed in V+2J selection by comparing the inclusive shape with and without ctagSF in 2D:
    in nJets-JetsHT bins. Each sample/year has a different correction stored in the correctionlib format.
    Note: the correction  file in parameters/ctagSF_calibrationSF.json is uesd by default here,
    which was  derived for V+2J phase space. It may not be suitable for other analyses.
    '''
    ctagSF = params.jet_scale_factors.ctagSF[year]
    cset = correctionlib.CorrectionSet.from_file(ctagSF.Calib_file)

    corr = cset["ctagSF_norm_correction"]
    w = corr.evaluate(dataset, ak.to_numpy(njets), ak.to_numpy(jetsHt))

    return w


def sf_jet_puId(params, jets, year, njets):
    # The SF is applied only on jets passing the preselection (JetGood), pt < maxpt, and matched to a GenJet.
    # In other words the SF is not applied on jets not passing the Jet Pu ID SF.
    # We DON'T assume that the function is applied on JetGood, so we REAPPLY jetPUID selections to be sure.
    # We apply also the pt limit.
    jet_puId_cfg = params.object_preselection["Jet"]["puId"]

    pt = ak.to_numpy(ak.flatten(jets.pt))
    eta = ak.to_numpy(ak.flatten(jets.eta))
    puId = ak.to_numpy(ak.flatten(jets.puId))
    genJetId_mask = ak.flatten(jets.genJetIdx >= 0)

    # GenGet matching by index, needs some checkes
    cset = correctionlib.CorrectionSet.from_file(
        params.jet_scale_factors.jet_puId[year]["file"]
    )
    corr = cset[params.jet_scale_factors.jet_puId[year]["name"]]

    # Requiring jet < maxpt, passing the jetpuid WP and matched to a GenJet (with a genJet idx != -1)
    mask = (
        (pt < jet_puId_cfg["maxpt"]) & (puId >= jet_puId_cfg["value"]) & genJetId_mask
    )
    index = (np.indices(pt.shape)).flatten()[mask]
    sf = np.ones_like(pt, dtype=float)
    sfup = np.ones_like(pt, dtype=float)
    sfdown = np.ones_like(pt, dtype=float)
    w = corr.evaluate(eta[mask], pt[mask], "nom", jet_puId_cfg["wp"])
    wup = corr.evaluate(eta[mask], pt[mask], "up", jet_puId_cfg["wp"])
    wdown = corr.evaluate(eta[mask], pt[mask], "down", jet_puId_cfg["wp"])
    sf[index] = w
    sfup[index] = wup
    sfdown[index] = wdown

    sf_out = ak.prod(ak.unflatten(sf, njets), axis=1)
    sf_up_out = ak.prod(ak.unflatten(sfup, njets), axis=1)
    sf_down_out = ak.prod(ak.unflatten(sfdown, njets), axis=1)

    return sf_out, sf_up_out, sf_down_out


def sf_L1prefiring(events):
    '''Correction due to the wrong association of L1 trigger primitives (TP) in ECAL to the previous bunch crossing,
    also known as "L1 prefiring".
    The event weights produced by the latest version of the producer are included in nanoAOD starting from version V9.
    The function returns the nominal, up and down L1 prefiring weights.'''
    L1PreFiringWeight = events.L1PreFiringWeight

    return L1PreFiringWeight['Nom'], L1PreFiringWeight['Up'], L1PreFiringWeight['Dn']


def sf_pileup_reweight(params, events, year):
    puFile = params.pileupJSONfiles[year]['file']
    puName = params.pileupJSONfiles[year]['name']

    puWeightsJSON = correctionlib.CorrectionSet.from_file(puFile)

    nPu = events.Pileup.nTrueInt.to_numpy()
    sf = puWeightsJSON[puName].evaluate(nPu, 'nominal')
    sfup = puWeightsJSON[puName].evaluate(nPu, 'up')
    sfdown = puWeightsJSON[puName].evaluate(nPu, 'down')

    return sf, sfup, sfdown
