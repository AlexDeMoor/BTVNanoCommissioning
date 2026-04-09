# DY Scale Factor Workflow — Deep Dive

## Overview

The **Drell-Yan (DY) scale factor** workflow measures b-tagging and c-tagging scale factors using Z-boson events that decay to dilepton (ee or mumu) final states. The Z-boson provides a clean, high-purity sample; the recoiling jet(s) serve as the probe for tagger calibration.

**Processor class**: `NanoProcessor` in `src/BTVNanoCommissioning/workflows/ctag_DY_valid_sf.py`
**Registered name**: `CTAGDYValidSFProcessor` (alias imported in `workflows/__init__.py:33`)

### Registered Workflow Variants

| Workflow key | `selectionModifier` | Channel | Description |
|---|---|---|---|
| `ctag_DY_sf` | `"DYM"` | mu+mu- | Standard DY dimuon |
| `ectag_DY_sf` | `"DYE"` | e+e- | Standard DY dielectron |
| `2D_mu_DY_sf` | `"DYM_2D"` | mu+mu- | 2D discriminant binning (dimuon) |
| `2D_e_DY_sf` | `"DYE_2D"` | e+e- | 2D discriminant binning (dielectron) |

All four are instantiated via `functools.partial` with different `selectionModifier` values.

---

## Full Dependency Map

```
ctag_DY_valid_sf.py (NanoProcessor)
|
+-- utils/correction.py
|   +-- load_lumi(campaign)         -> LumiMask from correctionlib/coffea
|   +-- load_SF(year, campaign)     -> Dict of scale factor evaluators
|   +-- common_shifts(self, events) -> Applies JEC/JER, muon SS, electron SS, jet veto
|   +-- weight_manager(pruned_ev, SF_map, isSyst) -> coffea Weights object
|       +-- genWeight, PSWeight, LHEPdfWeight, LHEScaleWeight
|       +-- top pT reweighting (for TT samples)
|       +-- puwei()           [pileup reweighting]
|       +-- muSFs()           [muon ID/Iso SFs]
|       +-- eleSFs()          [electron ID/Reco SFs]
|       +-- btagSFs()         [b/c-tag SFs via UParTAK4BC]
|
+-- helpers/func.py
|   +-- update(events, collections)   -> Shallow-copy events with swapped collections
|   +-- dump_lumi(events, output)     -> Store run/lumi pairs for data
|   +-- PFCand_link(events, ...)      -> Link PFCandidates to selected jets
|   +-- flatten(ar)                   -> Flatten awkward arrays to 1D for histogramming
|   +-- uproot_writeable(events, ...) -> Prepare events dict for uproot output
|
+-- helpers/update_branch.py
|   +-- missing_branch(events, campaign) -> Patch missing/renamed NanoAOD branches
|       (derives btagDeepFlavB, CvL, CvB, PNet, RobustParT, UParTAK4 variants)
|
+-- utils/selection.py
|   +-- HLT_helper(events, triggers)     -> OR of HLT trigger paths
|   +-- jet_id(events, campaign)         -> Jet ID + pT/eta cuts (campaign-aware)
|   +-- mu_idiso(events, campaign)       -> Tight muon ID + PF isolation < 0.15
|   +-- ele_mvatightid(events, campaign) -> Electron MVA tight WP80 + eta gap veto
|   +-- MET_filters(events, campaign)    -> MET quality filters + ecalBadCalibFilter
|
+-- utils/histogramming/histogrammer.py
|   +-- histogrammer(...)  -> Build histogram dict from collection specs
|   +-- histo_writer(...)  -> Fill histograms with selected events + weights
|   |
|   +-- hist_helpers.py
|       +-- get_axes_collections()  -> Load axis definitions
|       +-- get_hist_collections()  -> Load histogram templates
|           |
|           +-- axes/common.py     -> Shared axis definitions (flav, syst, pt, eta, ...)
|           +-- histograms/common.py    -> Discriminator & jet input histograms
|           +-- histograms/fourvec.py   -> 4-vector histograms (pt/eta/phi/mass per object)
|           +-- histograms/dy.py        -> DY-specific: dilep mass, dr, lepton dxy/dz
|           +-- histograms/dy_2D.py     -> DY 2D: dilep ptratio, top/antitop pt
|
+-- utils/array_writer.py
|   +-- array_writer(...)  -> Write selected events to ROOT TTrees via uproot
|
+-- utils/AK4_parameters.py
    +-- correction_config   -> Per-campaign dict of correction file paths
        (LUM, JME, BTV, MUO, EGM, muonSS, electronSS, jetveto, etc.)
```

---

## Processing Pipeline (step by step)

### 1. Initialization (`__init__`)

```python
self.lumiMask = load_lumi(self._campaign)   # Golden JSON lumi mask
self.SF_map   = load_SF(self._year, self._campaign)  # All correction evaluators
```

`load_SF` reads `correction_config` from `AK4_parameters.py` for the given campaign, then loads:

| Key | Source | Content |
|---|---|---|
| `LUM` | correctionlib / custom ROOT | Pileup weights |
| `JME` | correctionlib / coffea JEC stack | JEC, JER, JES uncertainties |
| `MUO` | correctionlib `muon_Z.json.gz` | Muon ID/Iso scale factors |
| `EGM` | correctionlib `electron.json.gz` | Electron ID/Reco scale factors |
| `BTV` | correctionlib `btagging.json.gz` / `ctagging.json.gz` | b/c-tag SFs |
| `muonSS` | correctionlib `muon_scalesmearing.json.gz` | Muon scale & smearing |
| `electronSS` | correctionlib `electronSS_EtDependent.json.gz` | Electron scale & smearing |
| `jetveto` | correctionlib / ROOT | Jet veto maps |

### 2. Branch Patching (`missing_branch`)

**File**: `helpers/update_branch.py`

Before any selection, the events are patched to ensure consistent branch naming across NanoAOD versions:

- `fixedGridRhoFastjetAll` fallback from `Rho.fixedGridRhoFastjetAll`
- Derives composite discriminants when individual sub-scores exist but the combined score doesn't:
  - `btagDeepFlavB` = b + bb + lepb
  - `btagDeepFlavC` from CvL or CvB
  - `btagDeepFlavCvL` / `btagDeepFlavCvB` from C and B scores
  - `btagPNetCvNotB` from PNet scores
  - `btagRobustParTAK4CvNotB`
  - `btagUParTAK4BvC`, `btagUParTAK4HFvLF` + pt-binned variants
  - `btagUParTAK42Dbin` + pt-binned variants (using `btag_wp_dict` from `selection.py`)
- `METFixEE2017` override for Run 2 2017
- `PuppiMET.MetUnclustEnUpDeltaX/Y` from `ptUnclusteredUp`

### 3. Common Shifts (`common_shifts`)

**File**: `utils/correction.py:2974`

Produces a list of `(collection_dict, shift_name)` tuples. The nominal entry has `shift_name=None`; systematic variations have names like `"JESUp"`, `"JERDown"`, etc.

**Shift chain**:
1. **JME shifts** (if configured): Applies JEC/JER corrections to jets and MET. For systematic variations (`isSyst="all"` or `"JERC_split"`), adds JES/JER up/down shifts.
2. **Muon corrections**: Rochester (Run 2) or Muon Scale & Smearing (Run 3) applied to `events.Muon`.
3. **Electron corrections**: Electron scale & smearing applied to `events.Electron`.
4. **Jet veto**: Removes events containing jets in the veto map region.

The processor then loops over shifts via:
```python
processor.accumulate(
    self.process_shift(update(vetoed_events, collections), name)
    for collections, name in shifts
)
```

### 4. Event Selection (`process_shift`)

#### 4a. Trigger Selection

| Modifier | HLT Path |
|---|---|
| `DYM*` | `Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8` |
| `DYE*` | `Ele23_Ele12_CaloIdL_TrackIdL_IsoVL` |

Applied via `HLT_helper` which ORs the trigger bits and validates paths exist in the dataset.

#### 4b. MET Filters

`MET_filters(events, campaign)` applies standard quality flags from the `met_filters` dict (campaign-specific), plus a custom `ecalBadCalibFilter` veto for runs 362433-367144.

#### 4c. Lepton Selection

| Object | Cuts |
|---|---|
| **Muons** (`dilep_mu`) | pT > 12 GeV, tight ID, PF RelIso04 < 0.15, \|eta\| < 2.4 |
| **Electrons** (`dilep_ele`) | pT > 15 GeV, MVA tight WP80, supercluster eta gap veto (\|etaSC\| not in [1.4442, 1.566]) |

#### 4d. Dilepton Requirements

- At least one positive and one negative charge lepton of the primary flavor
- At least 2 same-flavor leptons total (`req_dilep_chrg`)
- Zero leptons of the other flavor (`req_otherdilep_chrg`) — ensures flavor exclusivity
- **2D variants only**: Leading lepton pT > 20 GeV (mu) or > 25 GeV (e)

#### 4e. Z-boson Mass Window

```
81 < m(l+l-) < 101 GeV   AND   pT(ll) > 15 GeV
```

#### 4f. Jet Selection

- `jet_id(events, campaign, min_pt=25)`: Campaign-aware jet ID with pT > 25 GeV, |eta| < 2.5
- Jets must be isolated from both the positive and negative lepton: deltaR(jet, lepton) > 0.4
- At least 1 jet passing all cuts

#### 4g. Combined Event Mask

```python
event_level = req_lumi & req_trig & req_dilep & req_dilepmass & req_jets & req_metfilter
```

If no events survive, an empty output (optionally with blank arrays) is returned immediately.

### 5. Selected Object Construction

After applying `event_level`, the workflow builds a pruned event collection:

| Field | Content |
|---|---|
| `SelJet` | All jets passing selection (leading jet used for histograms) |
| `posl` / `negl` | Positive / negative charge lepton |
| `MuonPlus/Minus` or `ElectronPlus/Minus` | Channel-specific aliases |
| `SelMuon` or `SelElectron` | Both leptons combined as 2-column array |
| `dilep` | Z-boson candidate (4-vector sum of pos + neg lepton) |
| `njet` | Jet multiplicity |
| `dr_mu1jet` / `dr_mu2jet` | deltaR between each lepton and leading jet |
| `PFCands` | PF candidates linked to the selected jet (if available) |

### 6. Weights (`weight_manager`)

**File**: `utils/correction.py:3099`

Builds a `coffea.analysis_tools.Weights` object with `storeIndividual=True`. Applied weights (MC only):

| Weight | Description |
|---|---|
| `genweight` | Generator-level event weight |
| `PS ISR/FSR` | Parton shower initial/final state radiation weights |
| `PDF` | LHE PDF weight variations |
| `scalevar` | LHE scale (muR/muF) variations |
| `ttbar_weight` | Top pT reweighting (TT samples only) |
| `pileup` | Pileup reweighting from `LUM` correctionlib |
| `muon ID/Iso` | Muon scale factors (if `MUO` in SF_map and `SelMuon` exists) |
| `electron ID/Reco` | Electron scale factors (if `EGM` in SF_map and `SelElectron` exists) |
| `b/c-tag` | UParTAK4BC b/c-tagging SFs (if `ctag` or `btag` in SF_map) |

Systematics are stored as `weights.variations` (up/down for each weight).

### 7. Histogram Output (`histogrammer` + `histo_writer`)

#### Histogram Collections Loaded

The DY workflow requests these histogram collections:
- `"common"` — Always loaded
- `"fourvec"` — Always loaded
- `"DY"` — Always loaded
- `"DY_2D"` — Only for 2D selection modifier variants

With `obj_list=["posl", "negl", "dilep", "jet0"]` and `include_m=isMu`.

#### Common Histograms (`histograms/common.py`)

- **Counters**: `njet`, `npv`
- **Discriminator scores**: All `btag*` discriminants found in jet fields (per-jet indexed: `btagDeepFlavB_0`, `btagPNetCvL_0`, `btagUParTAK4HFvLF_0`, etc.)
- **Jet input variables**: DeepJet/DeepCSV training inputs (track pT, IP significance, etc.) for jets present in the data

#### Four-Vector Histograms (`histograms/fourvec.py`)

For each object in `obj_list`:
- **Jets** (`jet0`): `jet0_pt`, `jet0_eta`, `jet0_phi`, `jet0_mass` — with gen-flavor axis
- **Leptons** (`posl`, `negl`, `dilep`): `posl_pt`, `posl_eta`, `posl_phi`, `negl_pt`, etc. — without flavor axis

#### DY Histograms (`histograms/dy.py`)

| Histogram | Axes | Description |
|---|---|---|
| `dilep_mass` | syst, mass(50-100 GeV) | Z-boson invariant mass |
| `dr_poslnegl` | syst, dr | deltaR between positive and negative lepton |
| `dr_posljet` | syst, flav, dr | deltaR between positive lepton and jet |
| `dr_negljet` | syst, flav, dr | deltaR between negative lepton and jet |
| `posl_dxy` | syst, dxy | Positive lepton transverse IP |
| `posl_dz` | syst, dz | Positive lepton longitudinal IP |
| `negl_dxy` | syst, dxy | Negative lepton transverse IP |
| `negl_dz` | syst, dz | Negative lepton longitudinal IP |
| `posl_pfRelIso04_all` | syst, iso | Positive lepton isolation (**muon channel only**, `include_m=True`) |
| `negl_pfRelIso04_all` | syst, iso | Negative lepton isolation (**muon channel only**) |

#### DY 2D Histograms (`histograms/dy_2D.py`)

Only loaded for `DYM_2D` / `DYE_2D` modifiers:

| Histogram | Axes | Description |
|---|---|---|
| `dilep_ptratio` | syst, flav, ptratio | pT(Z) / pT(jet) ratio |
| `top_pt` | syst, jpt | Top quark pT (for TT background in DY region) |
| `antitop_pt` | syst, jpt | Anti-top quark pT |

#### Histogram Filling (`histo_writer`)

The `histo_writer` function loops over all systematics and fills each histogram. Key behavior:

- **Gen-flavor** is computed from `hadronFlavour` + `partonFlavour` (MC) or set to 0 (data)
- **Tagger score histograms** use `weights.partial_weight(exclude=exclude_btv)` to avoid double-counting b-tag SFs
- **Lepton/dilep histograms** use the full event weight
- **DY-specific fills**: `dr_poslnegl`, `dr_posljet`, `dr_negljet`, `dilep_pt/eta/phi/mass`, `dilep_ptratio`

### 8. Array Output (`array_writer`)

When `--isArray` is passed, selected events are written to ROOT files via uproot. The writer:

1. Attaches per-event weights (nominal + individual + systematic variations)
2. Computes new fields added during processing (`SelJet`, `posl`, `negl`, `dilep`, etc.)
3. Applies kinematic-only filtering for specified objects
4. Writes `Events` TTree + `TotalEventCount` + `TotalEventWeight` metadata

---

## Axis Definitions Reference (`axes/common.py`)

| Key | Type | Range | Label |
|---|---|---|---|
| `flav` | IntCategory | [0,1,4,5,6] | Gen-flavour (0=light, 1=unmatched, 4=c, 5=b, 6=gluon) |
| `syst` | StrCategory (growth) | dynamic | Systematic label |
| `pt` | Regular(60) | 0-300 GeV | Transverse momentum |
| `jpt` | Regular(300) | 0-3000 GeV | Jet pT (extended range) |
| `eta` | Regular(25) | -2.5 to 2.5 | Pseudorapidity |
| `phi` | Regular(30) | -3 to 3 | Azimuthal angle |
| `mass` | Regular(50) | 0-300 GeV | Mass |
| `dr` | Regular(20) | 0-8 | Delta-R |
| `dxy` | Regular(40) | -0.05 to 0.05 cm | Transverse impact parameter |
| `dz` | Regular(40) | -0.01 to 0.01 cm | Longitudinal impact parameter |
| `iso` | Regular(30) | 0-0.05 | Relative isolation |
| `ptratio` | Regular(50) | 0-1 | pT ratio |
| `npv` | Integer | 0-100 | Number of primary vertices |
| `n` | Integer | 0-10 | Object count |

---

## How to Run

```bash
# Standard DY dimuon
python runner.py --workflow ctag_DY_sf \
  --json metadata/test_bta_run3.json \
  --limit 1 --executor iterative \
  --year 2023 --campaign Summer23

# DY dielectron
python runner.py --workflow ectag_DY_sf \
  --json metadata/test_bta_run3.json \
  --limit 1 --executor iterative \
  --year 2023 --campaign Summer23

# 2D dimuon variant
python runner.py --workflow 2D_mu_DY_sf \
  --json metadata/test_bta_run3.json \
  --limit 1 --executor iterative \
  --year 2023 --campaign Summer23

# With systematics
python runner.py --workflow ctag_DY_sf \
  --json metadata/test_bta_run3.json \
  --limit 1 --executor iterative \
  --year 2023 --campaign Summer23 \
  --isSyst all

# With array output (ROOT TTrees)
python runner.py --workflow ctag_DY_sf \
  --json metadata/test_bta_run3.json \
  --limit 1 --executor iterative \
  --year 2023 --campaign Summer23 \
  --isArray
```

---

## Key File Locations

| File | Role |
|---|---|
| `src/BTVNanoCommissioning/workflows/ctag_DY_valid_sf.py` | Processor implementation |
| `src/BTVNanoCommissioning/workflows/__init__.py` | Workflow registration (lines 130-134) |
| `src/BTVNanoCommissioning/utils/correction.py` | `load_lumi`, `load_SF`, `common_shifts`, `weight_manager` |
| `src/BTVNanoCommissioning/utils/selection.py` | `HLT_helper`, `jet_id`, `mu_idiso`, `ele_mvatightid`, `MET_filters` |
| `src/BTVNanoCommissioning/helpers/func.py` | `update`, `dump_lumi`, `PFCand_link`, `flatten` |
| `src/BTVNanoCommissioning/helpers/update_branch.py` | `missing_branch` — NanoAOD branch patching |
| `src/BTVNanoCommissioning/utils/histogramming/histogrammer.py` | `histogrammer`, `histo_writer` |
| `src/BTVNanoCommissioning/utils/histogramming/hist_helpers.py` | Collection registry |
| `src/BTVNanoCommissioning/utils/histogramming/axes/common.py` | Axis definitions |
| `src/BTVNanoCommissioning/utils/histogramming/histograms/dy.py` | DY histogram templates |
| `src/BTVNanoCommissioning/utils/histogramming/histograms/dy_2D.py` | DY 2D histogram templates |
| `src/BTVNanoCommissioning/utils/histogramming/histograms/common.py` | Discriminator + jet input histograms |
| `src/BTVNanoCommissioning/utils/histogramming/histograms/fourvec.py` | 4-vector histograms |
| `src/BTVNanoCommissioning/utils/array_writer.py` | ROOT TTree output writer |
| `src/BTVNanoCommissioning/utils/AK4_parameters.py` | Per-campaign correction file paths |
| `src/BTVNanoCommissioning/helpers/definitions.py` | Tagger input variable definitions |
