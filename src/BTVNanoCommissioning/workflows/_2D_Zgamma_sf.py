import os
import collections, awkward as ak, numpy as np
from coffea import processor
from coffea.analysis_tools import Weights

from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    weight_manager,
    common_shifts,
)
from BTVNanoCommissioning.helpers.func import update, dump_lumi, PFCand_link, flatten
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.histogramming.histogrammer import (
    histogrammer,
    histo_writer,
)
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    gamma_mvatightid,
    MET_filters,
    mu_loose,
    ele_loose,
    btag_wp,
)


class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2024",
        campaign="Summer24Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
        selectionModifier="ZG",
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        ## Load corrections
        self.SF_map = load_SF(self._year, self._campaign)
        self.selMod = selectionModifier

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        events = missing_branch(events, f"{self._year}_{self._campaign}")
        vetoed_events, shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(vetoed_events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")

        triggers = ["Photon45EB_TightID_TightIso"]

        output = {}
        if not self.noHist:
            output = histogrammer(
                events.Jet.fields,
                # obj_list=["photon", "jet0", "jet1"],
                obj_list=["photon", "jet0", "jet1", "MET"],
                # obj_list=[],
                hist_collections=["common", "fourvec", "ZGamma_2D"],
                # include_nmujet=False,
                # include_nsoftmu=False,
                # include_osss=False,
                # cutbased=("cutbased" in self.selMod),
                # year=self._year,
                # campaign=self._campaign,
                njet=1,
            )

        if shift_name is None:
            output["other_sumw"] = len(events) if isRealData else np.sum(events.genWeight)

        ####################
        #    Selections    #
        ####################

        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        ## HLT
        req_trig = HLT_helper(events, triggers)

        ## Photon cuts
        id_gamma = events.Photon[
            (events.Photon.pt > 45) & gamma_mvatightid(events, self._campaign) & (events.Photon.pixelSeed == 0) & (abs(events.Photon.eta) < 1.4442) & (events.Photon.trkSumPtSolidConeDR04 < 0.025 * events.Photon.pt)
        ]
        req_gamma = ak.count(id_gamma.pt, axis=1) == 1

        event_photon = ak.pad_none(id_gamma, 1, axis=1)
        event_photon = event_photon[:, 0]
    
        # Now veto leptons
        # first select loose muons
        events.Muon = events.Muon[
            (events.Muon.pt > 15) & mu_loose(events, self._campaign)
        ]
        events.Muon = ak.pad_none(events.Muon, 1, axis=1)
        req_muon = ak.count(events.Muon.pt, axis=1) == 0

        events.Electron = events.Electron[
            (events.Electron.pt > 15) & ele_loose(events, self._campaign)
        ]
        events.Electron = ak.pad_none(events.Electron, 1, axis=1)
        req_ele = ak.count(events.Electron.pt, axis=1) == 0

        # iso_lep = ak.pad_none(iso_lep, 1, axis=1)
        # iso_lep = iso_lep[:, 0]
        # sub_lep = events.Muon[
        #     (events.Muon.pt < 30) &  (abs(events.Muon.eta) < 2.4) & (events.Muon.tightId > 0.5)
        # ]
        # sub_lep = ak.pad_none(sub_lep, 1, axis=1)
        # sub_lep = sub_lep[:, 0]

        ## Jet cuts
        jet_sel = ak.fill_none(
            jet_id(events, self._campaign, min_pt=25) & (events.Jet.pt > 25) & (abs(events.Jet.eta) < 2.5)
            & (ak.all(events.Jet.metric_table(id_gamma) > 0.5, axis=2)) & (ak.all(events.Jet.metric_table(events.Muon) > 0.5, axis=2)) & (ak.all(events.Jet.metric_table(events.Electron) > 0.5, axis=2)),
            False,
            axis=-1,
        )
        event_jet = events.Jet[jet_sel]
        event_jet = ak.pad_none(event_jet, 2, axis=1)
        nseljet = ak.count(event_jet.pt, axis=1)
        # req_base_jets = (nseljet >= 1) & (nseljet <= 3)
        req_jets = nseljet == 2



        ## Soft Muon cuts
        soft_muon = events.Muon[
            (events.Muon.pt < 25)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.tightId > 0.5)
            & (events.Muon.pfRelIso04_all > 0.25)
        ]
        # soft_muon_tight = soft_muon[
        #     (abs(soft_muon.dxy / soft_muon.dxyErr) > dxySigcut)
        #     & (soft_muon.jetIdx != -1)
        # ]
        req_softmu = ak.count(soft_muon.pt, axis=1) >= 1
        # req_softmu_tight = ak.count(soft_muon_tight.pt, axis=1) >= 1
        req_softmu_pt = ak.count(soft_muon[soft_muon.pt > 5.0].pt, axis=1) >= 1

        mujetsel = ak.fill_none(
            (
                (ak.all(event_jet.metric_table(soft_muon) <= 0.4, axis=2))
                & ((event_jet.muonIdx1 != -1) | (event_jet.muonIdx2 != -1))
                & ((event_jet.muEF + event_jet.neEmEF) < 1.0)
                & (event_jet.pt > 20)
                & ((event_jet.pt / event_jet.E) > 0.03)
            ),
            False,
            axis=-1,
        )
        soft_muon = ak.pad_none(soft_muon, 1, axis=1)
        mu_jet = event_jet[mujetsel]
        # otherjets = event_jet[~mujetsel]
        req_mujet = ak.num(mu_jet.pt, axis=1) >= 1
        mu_jet = ak.pad_none(mu_jet, 1, axis=1)

        # req_1mujet = ak.count(mu_jet.pt, axis=1) == 1
        # req_1otherjet = ak.count(otherjets.pt, axis=1) == 1

        # do a tag and probe selection. If the first jet is a muon jet, select the second jet regardless of what the second jet is. In addition, if the second jet is a muon jet, select the first jet regardless of what the first jet is. This is to keep more events for the SF measurement.
        jet0_in_mu_jet = ak.any(event_jet[:, 0].metric_table(mu_jet) == 0, axis=1)
        jet1_in_mu_jet = ak.any(event_jet[:, 1].metric_table(mu_jet) == 0, axis=1)
        selected_jet = ak.where(jet0_in_mu_jet, event_jet[:, 1], event_jet[:, 0])

        selected_jet = ak.pad_none(selected_jet, 1, axis=1)
    

        # jet energy fraction cuts
        muNeEmSum_sel = ak.fill_none(
            (event_jet.muEF + event_jet.neEmEF) < 1.0,
            False,
            axis=-1,
        )
        # req_muNeEmSum = ak.num(event_jet[muNeEmSum_sel].pt, axis=1) == 1
        muEF_sel = ak.fill_none(
            (event_jet.muEF < 0.5),
            False,
            axis=-1,
        )
        # req_muEF = ak.num(event_jet[muEF_sel].pt, axis=1) >= 1


        # correct jet pt with pnet
        # event_jet.pt = event_jet.pt * event_jet.PNetRegPtRawCorr

        # Other cuts
        # Ratio between soft muon pt and muon jet pt
        # req_pTratio = (soft_muon[:, 0].pt / mu_jet[:, 0].pt) < muonpTratioCut
        # idx = np.where(iso_lep.jetIdx == -1, 0, iso_lep.jetIdx)
        ## Additional cut to reject QCD events, used in BTV-20-001
        # req_QCDveto = (
        #     (iso_lep.pfRelIso04_all < 0.05)
        # & (abs(iso_lep.dz) < isolepdz)
        # & (abs(iso_lep.dxy) < isolepdxy)
        # & (iso_lep.sip3d < isolepsip3d)
        # & (
        #     iso_lep.pt
        #     / ak.firsts(
        #         events.Jet[
        #             (events.Jet.muonIdx1 == iso_lepindx)
        #             | ((events.Jet.muonIdx2 == iso_lepindx))
        #         ].pt
        #     )
        #     > 0.75
        # )
        # )

        ## MET and Transverse W mass
        # iso_lep_trans = ak.zip(
        #     {
        #         "pt": iso_lep.pt,
        #         "eta": ak.zeros_like(iso_lep.pt),
        #         "phi": iso_lep.phi,
        #         "mass": iso_lep.mass,
        #     },
        #     with_name="PtEtaPhiMLorentzVector",
        # )

        MET = ak.zip(
            {
                "pt": events.PuppiMET.pt,
                "eta": ak.zeros_like(events.PuppiMET.pt),
                "phi": events.PuppiMET.phi,
                "mass": ak.zeros_like(events.PuppiMET.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        req_metfilter = MET_filters(events, self._campaign)
        event_jet_0 = ak.pad_none(event_jet, 1, axis=1)
        event_jet_0 = event_jet_0[:, 0]
        
        event_jet_1 = ak.pad_none(event_jet, 2, axis=1)
        event_jet_1 = event_jet_1[:, 1]

        req_met_pt = MET.pt < 50.0

        # jetmet_dphi = abs(event_jet_0.delta_phi(MET))
        # req_jetmet_dphi = jetmet_dphi > 1.0
        # metTrkmet_dphi = abs(MET.delta_phi(events.TrkMET))
        # req_metTrkmet_dphi = metTrkmet_dphi < 1.0  # & (metTrkmet_dphi >= 0.5)

        # Z cuts
        # zmasscut = 55  # before 55
        zmasscut = 70  # before 55
        # zmasscut_tight = 55
        zptcut = 50

        Zcand = event_jet_0 + event_jet_1
        Zmass = Zcand.mass
        Zpt = Zcand.pt
        req_ZpT = Zpt > zptcut
        req_mZ = Zmass > zmasscut
        # req_mZ_max180 = Zmass < 125
        req_mZ_max180 = Zmass < 110
        # req_mZ_min55 = Zmass > zmasscut_tight

        # req for pt ratio of leading photon and W pt
        photonz_ptratio = event_photon.pt / Zpt
        # req_jZpTratio = (photonz_ptratio < 2) & (photonz_ptratio > 0.5)
        # req_jZpTratio = (photonz_ptratio < 1.5) & (photonz_ptratio > 0.5)
        req_jZpTratio = (photonz_ptratio < 1.25) & (photonz_ptratio > 0.75)

        # mask on distance between W_phi and photon_phi abs(deltaPhi(ak4jets[0].phi, Vboson.phi) > 2 (if tight sel)
        photonz_dphi = abs(event_photon.delta_phi(Zcand))
        # req_dphi_Zphoton = photonz_dphi > 2.0
        req_dphi_Zphoton = photonz_dphi > 2.5

        # # delta Phi between leading jet and lepton
        # jetl_dphi = abs(event_jet_0.delta_phi(iso_lep))
        # req_jetl_dphi = jetl_dphi < 2.0


        # derive some jet-jet variables
        dr_jet_jet = event_jet_0.delta_r(event_jet_1)
        dphi_jet_jet = event_jet_0.delta_phi(event_jet_1)
        deta_jet_jet = abs(event_jet_0.eta - event_jet_1.eta)

        dr_jet0_photon = event_jet_0.delta_r(event_photon)
        dphi_jet0_photon = event_jet_0.delta_phi(event_photon)
        deta_jet0_photon = abs(event_jet_0.eta - event_photon.eta)
        dr_jet1_photon = event_jet_1.delta_r(event_photon)
        dphi_jet1_photon = event_jet_1.delta_phi(event_photon)
        deta_jet1_photon = abs(event_jet_1.eta - event_photon.eta)

        ptratio_jet_jet = event_jet_0.pt / event_jet_1.pt
        ptratio_jet0_photon = event_jet_0.pt / event_photon.pt
        ptratio_jet1_photon = event_jet_1.pt / event_photon.pt

        photon_trkSumPtSolidConeDR04 = event_photon.trkSumPtSolidConeDR04
        photon_hoe = event_photon.hoe

        # ==This is the manual calculation for transverse mass==
        """
        dphi = iso_lep.phi-events.PuppiMET.phi
        dphi = np.where(dphi<np.pi,dphi+2*np.pi,dphi)
        dphi = np.where(dphi>np.pi,dphi-2*np.pi,dphi)
        trans = np.sqrt(2*iso_lep.pt*events.PuppiMET.pt*(1-np.cos(dphi)))
        """

        event_level = (
            req_trig
            & req_lumi
            & req_metfilter
            & req_gamma
            & req_jets
            # & req_muon
            # & req_ele
            & req_met_pt
            & req_mZ
            & req_mZ_max180
            & req_ZpT
            & req_jZpTratio
            & req_dphi_Zphoton

            & req_softmu
            # & req_softmu_pt
            # & req_mujet
            # & req_1mujet
            # & req_1otherjet
            # & req_muNeEmSum
            # & req_muEF
        )

        event_level = ak.fill_none(event_level, False)
        if len(events[event_level]) == 0:
            if self.isArray:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    None,
                    ["nominal"],
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}

        ####################
        # Selected objects #
        ####################

        sg = id_gamma[event_level]
        zm = Zmass[event_level]
        zp = Zpt[event_level]
        # sjets = event_jet[event_level]
        sjets = selected_jet[event_level]
        smet = MET[event_level]
        # smuon_jet = mu_jet[event_level]
        # sotherjets = otherjets[event_level]
        # sdilep = dilep_mass[event_level]
        # nsoftmu = ak.count(ssmu.pt, axis=1)
        # smuon_jet = sjets[:, 0]
        # ssmu = ssmu[:, 0]
        # sz = shmu + ssmu
        sz = Zcand[event_level]
        # sw = shmu + smet

        # osss = shmu.charge * ssmu.charge * -1
        # ossswrite = shmu.charge * ssmu.charge * -1

        njet = ak.count(sjets.pt, axis=1)
        # Find the PFCands associate with selected jets, search from jetindex->JetPFCands->PFCand
        # if "PFCands" in events.fields:
        #     spfcands = PFCand_link(events, event_level, jetindx)

        # Keep the structure of events and pruned the object size
        pruned_ev = events[event_level]
        pruned_ev["SelJet"] = sjets
        pruned_ev["SelPhoton"] = sg
        # pruned_ev["OtherJets"] = sotherjets
        pruned_ev["MET_pt"] = smet.pt
        pruned_ev["MET"] = smet
        pruned_ev["njet"] = njet
        pruned_ev["z_mass"] = zm
        pruned_ev["z_pt"] = zp

        b_jet_mask = btag_wp(
            event_jet[event_level], self._year, self._campaign, "UParTAK4", "b", "M"
        )
        c_jet_mask = btag_wp(
            event_jet[event_level], self._year, self._campaign, "UParTAK4", "c", "M"
        )
        pruned_ev["nbjet"] = ak.count(event_jet[event_level].pt[b_jet_mask], axis=1)
        pruned_ev["ncjet"] = ak.count(event_jet[event_level].pt[c_jet_mask], axis=1)
        # pruned_ev["dilep_mass"] = sdilep.mass
        # pruned_ev["dilep_pt"] = sdilep.pt
        pruned_ev["photonz_dphi"] = photonz_dphi[event_level]
        # pruned_ev["z_mt"] = event_Wmt[event_level]
        pruned_ev["z_hadmass"] = zm
        # pruned_ev["jetmet_dphi"] = jetmet_dphi[event_level]
        # pruned_ev["metTrkmet_dphi"] = metTrkmet_dphi[event_level]
        # pruned_ev["jetl_dphi"] = jetl_dphi[event_level]
        # if "PFCands" in events.fields:
        #     pruned_ev.PFCands = spfcands

        # # Add custom variables
        # pruned_ev["dr_mujet_softmu"] = ssmu.delta_r(smuon_jet)
        # pruned_ev["dr_mujet_lep1"] = shmu.delta_r(smuon_jet)
        # pruned_ev["dr_lep1_softmu"] = shmu.delta_r(ssmu)
        # pruned_ev["soft_l_ptratio"] = ssmu.pt / smuon_jet.pt
        # pruned_ev["l1_ptratio"] = shmu.pt / smuon_jet.pt
        # pruned_ev["MuonJet_beta"] = smuon_jet.pt / smuon_jet.E
        # pruned_ev["MuonJet_muneuEF"] = smuon_jet.muEF + smuon_jet.neEmEF
        # pruned_ev["jetw_ptratio"] = jetw_ptratio[event_level]

        # print (pruned_ev.SelJet.fields)

        if "hadronFlavour" in pruned_ev.SelJet.fields:
            isRealData = False
            genflavor = ak.values_astype(
                pruned_ev.SelJet.hadronFlavour
                + 1
                * (
                    (pruned_ev.SelJet.partonFlavour == 0)
                    & (pruned_ev.SelJet.hadronFlavour == 0)
                ),
                int,
            )
            # if "MuonJet" in pruned_ev.fields:
            #     smflav = ak.values_astype(
            #         1
            #         * (
            #             (pruned_ev.MuonJet.partonFlavour == 0)
            #             & (pruned_ev.MuonJet.hadronFlavour == 0)
            #         )
            #         + pruned_ev.MuonJet.hadronFlavour,
            #         int,
            #     )
        else:
            isRealData = True
            genflavor = ak.ones_like(pruned_ev.SelJet.pt, dtype=int)
            # if "MuonJet" in pruned_ev.fields:
            #     smflav = ak.ones_like(pruned_ev.MuonJet.pt, dtype=int)

        # # Store masks for additional cuts
        # # pruned_ev["mask_njets"] = req_jets[event_level]
        # pruned_ev["mask_mujet"] = req_mujet[event_level]
        # pruned_ev["mask_muNeEmSum"] = req_muNeEmSum[event_level]
        # pruned_ev["mask_muEF"] = req_muEF[event_level]
        # pruned_ev["mask_jetl_dphi"] = req_jetl_dphi[event_level]
        # pruned_ev["mask_dilepveto"] = req_dilepveto[event_level]
        # pruned_ev["mask_smj_ptratio"] = req_pTratio[event_level]
        # pruned_ev["mask_met_pt"] = req_met_pt[event_level]
        # pruned_ev["mask_jetmet_dphi"] = req_jetmet_dphi[event_level]
        # pruned_ev["mask_metTrkmet_dphi"] = req_metTrkmet_dphi[event_level]
        # pruned_ev["req_mtw_max120"] = req_mtw_max120[event_level]
        # pruned_ev["req_mtw_min55"] = req_mtw_min55[event_level]
        # pruned_ev["mask_wpt"] = req_WpT[event_level]
        # pruned_ev["mask_softmu_tight"] = req_softmu_tight[event_level]
        # pruned_ev["mask_softmu_pt"] = req_softmu_pt[event_level]
        # pruned_ev["mask_jWpTratio"] = req_jWpTratio[event_level]
        # pruned_ev["mask_dphi_Wjet"] = req_dphi_Wjet[event_level]

        ####################
        # Weight & Geninfo #
        ####################

        weights = weight_manager(pruned_ev, self.SF_map, self.isSyst)

        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]
        exclude_btv = [
            "DeepCSVC",
            "DeepCSVB",
            "DeepJetB",
            "DeepJetC",
        ]  # exclude b-tag SFs for btag inputs

        # Configure histograms
        if not self.noHist:
            output = histo_writer(
                pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
            )

        ####################
        #  Fill histogram  #
        ####################

        for syst in systematics:
            if self.isSyst == False and syst != "nominal":
                break
            if self.noHist:
                break
            weight = (
                weights.weight()
                if syst == "nominal" or syst == shift_name
                else weights.weight(modifier=syst)
            )
            syst = np.full(len(weight), syst)
            for histname, h in output.items():
                # print (histname)
                if (
                    "Deep" in histname
                    and "btag" not in histname
                    and histname in events.Jet.fields
                ):
                    h.fill(
                        syst,
                        flatten(genflavor),
                        flatten(ak.broadcast_arrays(sjets["pt"])[0]),
                        flatten(sjets[histname]),
                        weight=flatten(
                            ak.broadcast_arrays(
                                weights.partial_weight(exclude=exclude_btv), sjets["pt"]
                            )[0]
                        ),
                    )
                # elif (
                #     "PFCands" in events.fields
                #     and "PFCands" in histname
                #     and histname.split("_")[1] in events.PFCands.fields
                # ):
                #     h.fill(
                #         syst,
                #         # flatten(ak.broadcast_arrays(smflav, spfcands["pt"])[0]),
                #         # flatten(ak.broadcast_arrays(osss, spfcands["pt"])[0]),
                #         flatten(spfcands[histname.replace("PFCands_", "")]),
                #         weight=flatten(
                #             ak.broadcast_arrays(
                #                 weights.partial_weight(exclude=exclude_btv),
                #                 spfcands["pt"],
                #             )[0]
                #         ),
                #     )
                elif "jet_" in histname and "mu" not in histname:
                    h.fill(
                        syst,
                        flatten(genflavor),
                        # flatten(ak.broadcast_arrays(osss, sjets["pt"])[0]),
                        flatten(sjets[histname.replace("jet_", "")]),
                        weight=flatten(ak.broadcast_arrays(weight, sjets["pt"])[0]),
                    )
                # elif "hl_" in histname and histname.replace("hl_", "") in shmu.fields:
                #     h.fill(
                #         syst,
                #         osss,
                #         flatten(shmu[histname.replace("hl_", "")]),
                #         weight=weight,
                #     )
                # elif (
                #     "soft_l" in histname
                #     and histname.replace("soft_l_", "") in ssmu.fields
                # ):
                #     h.fill(
                #         syst,
                #         # smflav,
                #         # osss,
                #         flatten(ssmu[histname.replace("soft_l_", "")]),
                #         weight=weight,
                #     )
                # elif (
                #     "mujet_" in histname
                #     and histname.replace("mujet_", "") in smuon_jet.fields
                # ):
                #     h.fill(
                #         syst,
                #         smflav,
                #         osss,
                #         flatten(smuon_jet[histname.replace("mujet_", "")]),
                #         weight=weight,
                #     )
                # elif "btag" in histname and "Trans" not in histname:
                #     for i in range(2):
                #         if (
                #             not histname.endswith(str(i))
                #             or histname.replace(f"_{i}", "") not in smuon_jet.fields
                #         ):
                #             continue
                #         h.fill(
                #             syst,
                #             # flav=smflav,
                #             # osss=osss,
                #             discr=np.where(
                #                 smuon_jet[histname.replace(f"_{i}", "")] < 0,
                #                 -2,
                #                 smuon_jet[histname.replace(f"_{i}", "")],
                #             ),
                #             weight=weight,
                #             # weight=weights.partial_weight(exclude=exclude_btv),
                #         )
                #         if not isRealData and "btag" in self.SF_map.keys():
                #             h.fill(
                #                 syst=syst,
                #                 # flav=smflav,
                #                 # osss=osss,
                #                 discr=np.where(
                #                     smuon_jet[histname.replace(f"_{i}", "")] < 0,
                #                     -0.2,
                #                     smuon_jet[histname.replace(f"_{i}", "")],
                #                 ),
                #                 weight=weight,
                #             )
                # elif "btag" in histname and "Trans" in histname:
                #     if histname not in smuon_jet:
                #         continue
                #     for i in range(2):
                #         histname = histname.replace("Trans", "").replace(f"_{i}", "")
                #         h.fill(
                #             syst="noSF",
                #             flav=smflav,
                #             osss=osss,
                #             discr=1.0 / np.tanh(smuon_jet[histname]),
                #             weight=weights.partial_weight(exclude=exclude_btv),
                #         )

            output["njet"].fill(syst, njet, weight=weight)
            # output["nsoftmu"].fill(syst, osss, nsoftmu, weight=weight)
            output["npv"].fill(syst, pruned_ev.PV.npvsGood, weight=weight)
            # output["softlpt"].fill(syst, smflav, osss, ssmu.pt, weight=weight)
            # output["hl_ptratio"].fill(
            #     syst,
            #     genflavor[:, 0],
            #     osss=osss,
            #     ratio=shmu.pt / sjets[:, 0].pt,
            #     weight=weight,
            # )
            # output["soft_l_ptratio"].fill(
            #     syst,
            #     flav=smflav,
            #     osss=osss,
            #     ratio=ssmu.pt / smuon_jet.pt,
            #     weight=weight,
            # )
            # output["dr_lmujetsmu"].fill(
            #     syst,
            #     flav=smflav,
            #     osss=osss,
            #     dr=smuon_jet.delta_r(ssmu),
            #     weight=weight,
            # )
            # output["dr_lmujethmu"].fill(
            #     syst,
            #     flav=smflav,
            #     osss=osss,
            #     dr=smuon_jet.delta_r(shmu),
            #     weight=weight,
            # )
            # output["dr_hmusmu"].fill(
            #     syst,
            #     osss=osss,
            #     dr=shmu.delta_r(ssmu),
            #     weight=weight,
            # )
            # output["mujet_muneuEF"].fill(
            #     syst,
            #     smflav,
            #     osss=osss,
            #     muneuEF=flatten(pruned_ev["MuonJet_muneuEF"]),
            #     weight=weight,
            # )
            # output["dilep_pt"].fill(syst, osss, flatten(sz.pt), weight=weight)
            # output["dilep_eta"].fill(syst, osss, flatten(sz.eta), weight=weight)
            # output["dilep_phi"].fill(syst, osss, flatten(sz.phi), weight=weight)
            # output["dilep_mass"].fill(syst, osss, flatten(sz.mass), weight=weight)
            output["z_pt"].fill(syst, flatten(sz.pt), weight=weight)
            output["z_phi"].fill(syst, flatten(sz.phi), weight=weight)
            output["z_mass"].fill(syst, flatten(sz.mass), weight=weight)
            output["photonzdphi"].fill(
                syst, photonz_dphi[event_level], weight=weight
            )
            output["photonzptratio"].fill(
                syst, flatten(photonz_ptratio[event_level]), weight=weight
            )
            output["drjj"].fill(syst, dr_jet_jet[event_level], weight=weight)
            output["dphijj"].fill(syst, dphi_jet_jet[event_level], weight=weight)
            output["detajj"].fill(syst, deta_jet_jet[event_level], weight=weight)
            output["drj0photon"].fill(syst, dr_jet0_photon[event_level], weight=weight)
            output["dphij0photon"].fill(syst, dphi_jet0_photon[event_level], weight=weight)
            output["detaj0photon"].fill(syst, deta_jet0_photon[event_level], weight=weight)
            output["drj1photon"].fill(syst, dr_jet1_photon[event_level], weight=weight)
            output["dphij1photon"].fill(syst, dphi_jet1_photon[event_level], weight=weight)
            output["detaj1photon"].fill(syst, deta_jet1_photon[event_level], weight=weight)
            output["ptratiojj"].fill(syst, ptratio_jet_jet[event_level], weight=weight)
            output["ptratioj0photon"].fill(syst, ptratio_jet0_photon[event_level], weight=weight)
            output["ptratioj1photon"].fill(syst, ptratio_jet1_photon[event_level], weight=weight)

            output["MET_pt"].fill(syst, flatten(smet.pt), weight=weight)
            output["MET_phi"].fill(syst, flatten(smet.phi), weight=weight)
            # output["jetmet_dphi"].fill(
            #     syst, smflav, osss, flatten(jetmet_dphi[event_level]), weight=weight
            # )
            # output["metTrkmet_dphi"].fill(
            #     syst, osss, flatten(metTrkmet_dphi[event_level]), weight=weight
            # )
            # output["jetl_dphi"].fill(
            #     syst, smflav, osss, flatten(jetl_dphi[event_level]), weight=weight
            # )

        output["photonhoe"].fill(syst, flatten(photon_hoe[event_level]), weight=weight)
        output["photontrkSumPtSolidConeDR04"].fill(syst, flatten(photon_trkSumPtSolidConeDR04[event_level]/sg.pt), weight=weight)

        #######################
        #  Create root files  #
        #######################

        if self.isArray:
            array_writer(
                self, pruned_ev, events, weights, systematics, dataset, isRealData
            )

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
