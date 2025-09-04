from ..calibrator import Calibrator
import numpy as np
import awkward as ak
import cachetools
from pocket_coffea.lib.jets import jet_correction, met_correction_after_jec, load_jet_factory, jet_correction_clib
from pocket_coffea.lib.leptons import get_ele_scaled, get_ele_smeared

class JetsCalibrator(Calibrator):
    """
    This calibator applies the JEC to the jets and their uncertainties. 
    The set of calibrations to be applied is defined in the parameters file under the 
    `jets_calibration.collection` section.
    All the jet types that have apply_jec_MC or apply_jec_Data set to True will be calibrated.
    If the pT regression is requested for a jet type, it should be done by the JetsPtRegressionCalibrator, 
    this calibrator will raise an exception if configured to apply pT regression.
    """
    
    name = "jet_calibration_clib"
    has_variations = True
    isMC_only = False

    def __init__(self, params, metadata, **kwargs):
        super().__init__(params, metadata, **kwargs)
        self._year = metadata["year"]
        self.jet_calib_param = self.params.jets_calibration
        self.jet_calib_param_clib = self.params.jets_calibration_clib
        self.variations_params = self.params.jet_calibration.variations
        self.caches = [] 
        self.jets_calibrated = {}
        self.jets_calibrated_types = []
        # It is filled dynamically in the initialize method
        self.calibrated_collections = []

    def initialize(self, events):
        # Load the calibration of each jet type requested by the parameters
        for jet_type, jet_coll_name in self.jet_calib_param.collection[self.year].items():
            # Check if the collection is enabled in the parameters
            if self.isMC:
                if (self.jet_calib_param.apply_jec_MC[self.year][jet_type] == False):
                    # If the collection is not enabled, we skip it
                    continue
            else:
                if self.jet_calib_param.apply_jec_Data[self.year][jet_type] == False:
                    # If the collection is not enabled, we skip it
                    continue

            if jet_coll_name in self.jets_calibrated:
                # If the collection is already calibrated with another jet_type, raise an error for misconfiguration
                raise ValueError(f"Jet collection {jet_coll_name} is already calibrated with another jet type. " +
                                 f"Current jet type: {jet_type}. Previous jet types: {self.jets_calibrated[jet_coll_name]}")

            # Check the Pt regression is not requested for this jet type 
            # and in that case send a warning and skim them
            if self.isMC and self.jet_calib_param.apply_pt_regr_MC[self.year][jet_type]:
                print(f"WARNING: Jet type {jet_type} is requested to be calibrated with pT regression: " +
                                    "skipped by JetCalibrator. Please activate the JetsPtRegressionCalibrator.")
                continue
            if not self.isMC and self.jet_calib_param.apply_pt_regr_Data[self.year][jet_type]:
                print(f"WARNING: Jet type {jet_type} is requested to be calibrated with pT regression: " +
                                    "skipped by JetCalibrator. Please activate the JetsPtRegressionCalibrator.")
                continue

            # register the collection as calibrated by this calibrator
            self.calibrated_collections.append(jet_coll_name)

            cache = cachetools.Cache(np.inf)
            self.caches.append(cache)
            # self.jets_calibrated[jet_coll_name] = jet_correction(
            #     params=self.params,
            #     events=events,
            #     jets=events[jet_coll_name],
            #     factory=self.jme_factory,
            #     jet_type = jet_type,
            #     chunk_metadata={
            #         "year": self.metadata["year"],
            #         "isMC": self.metadata["isMC"],
            #         "era": self.metadata["era"] if "era" in self.metadata else None,
            #     },
            #     cache=cache
            # )
            self.jets_calibrated[jet_coll_name] = jet_correction_clib(
                events=events,
                chunk_metadata={
                    "year": self.metadata["year"],
                    "isMC": self.metadata["isMC"],
                    "era": self.metadata["era"] if "era" in self.metadata else None,
                },
                params=self.params,
                level="L1L2L3Res",
                algo=jet_type,
                jet_coll_name=jet_coll_name
            )
            
            # Add to the list of the types calibrated
            self.jets_calibrated_types.append(jet_type)

        # Prepare the list of available variations
        # For this we just read from the parameters
        available_jet_variations = []
        for jet_type in self.jet_calib_param.collection[self.year].keys():
            if jet_type not in self.jets_calibrated_types:
                # If the jet type is not calibrated, we skip it
                continue
            if jet_type in self.jet_calib_param.variations[self.year]:
                # If the jet type has variations, we add them to the list
                # of variations available for this calibrator
                for variation in self.jet_calib_param.variations[self.year][jet_type]:
                    available_jet_variations +=[
                        f"{jet_type}_{variation}Up",
                        f"{jet_type}_{variation}Down"
                    ]
                    # we want to vary independently each jet type
        self._variations = list(sorted(set(available_jet_variations)))  # remove duplicates


    def calibrate(self, events, orig_colls, variation, already_applied_calibrators=None):
        # The values have been already calculated in the initialize method
        # We just need to apply the corrections to the events
        out = {}
        if variation == "nominal" or variation not in self._variations:
            # If the variation is nominal or not in the list of variations, we return the nominal values
            for jet_coll_name, jets in self.jets_calibrated.items():
                out[jet_coll_name] = jets
        else:
            # get the jet type from the variation name
            variation_parts = variation.split("_")
            jet_type = variation_parts[0]
            if jet_type not in self.jet_calib_param.collection[self.year]:
                raise ValueError(f"Jet type {jet_type} not found in the parameters for year {self.year}.")
            # get the variation type from the variation name
            if variation.endswith("Up"):
                variation_type = "_".join(variation_parts[1:])[:-2]  # remove 'Up'
                direction = "up"
            elif variation.endswith("Down"):
                variation_type = "_".join(variation_parts[1:])[:-4]  # remove 'Down'
                direction = "down"
            else:
                raise ValueError(f"JET Variation {variation} is not recognized. It should end with 'Up' or 'Down'.")
           
            # get the jet collection name from the parameters
            jet_coll_name = self.jet_calib_param.collection[self.year][jet_type]
            if jet_coll_name not in self.jets_calibrated:
                raise ValueError(f"Jet collection {jet_coll_name} not found in the calibrated jets.")
            # Apply the variation to the jets
            if direction == "up":
                out[jet_coll_name] = self.jets_calibrated[jet_coll_name][variation_type].up
            elif direction == "down":
                out[jet_coll_name] = self.jets_calibrated[jet_coll_name][variation_type].down
            
        return out
