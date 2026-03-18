import hist as Hist


def get_histograms(axes, **kwargs):
    hists = {}

    hists[f"nbjet"] = Hist.Hist(axes["syst"], axes["n"], Hist.storage.Weight())
    hists[f"ncjet"] = Hist.Hist(axes["syst"], axes["n"], Hist.storage.Weight())
    # hists[f"z_mt"] = Hist.Hist(axes["syst"], axes["mt"], Hist.storage.Weight())
    hists[f"z_hadmass"] = Hist.Hist(axes["syst"], axes["hadmass"], Hist.storage.Weight())

    # channel = kwargs.get("channel", "mu")
    # hists[f"{channel}_pfRelIso04_all"] = Hist.Hist(
    #     axes["syst"], axes["iso"], Hist.storage.Weight()
    # )
    # hists[f"{channel}_dxy"] = Hist.Hist(
    #     axes["syst"], axes["dxy"], Hist.storage.Weight()
    # )
    # hists[f"{channel}_dz"] = Hist.Hist(axes["syst"], axes["dz"], Hist.storage.Weight())

    # hists["top_pt"] = Hist.Hist(
    #     axes["syst"], axes["jpt"], Hist.storage.Weight()
    # )
    # hists["antitop_pt"] = Hist.Hist(
    #     axes["syst"], axes["jpt"], Hist.storage.Weight()
    # )


    hists["z_mass"] = Hist.Hist(
        axes["syst"],
        # Hist.axis.Regular(55, 40, 150, name="mass", label=" $m_{qq}$ [GeV]"),
        Hist.axis.Regular(80, 40, 200, name="mass", label=" $m_{qq}$ [GeV]"),
        Hist.storage.Weight(),
    )
    hists["z_pt"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 300, name="pT", label=" $p_T^{qq}$ [GeV]"),
        Hist.storage.Weight(),
    )
    hists["z_phi"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 4, name="phi", label=" $\\phi_{qq}$ [deg]"),
        Hist.storage.Weight(),
    )


    hists["photonzptratio"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 3, name="photonzptratio", label=" $p_T^{\\gamma}/p_T^{qq}$"),
        Hist.storage.Weight(),
    )
    hists["photonzdphi"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 4, name="photonzdphi", label=" $\\Delta\\phi_{\\gamma qq}$"),
        Hist.storage.Weight(),
    )


    hists["drjj"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 5, name="drjj", label=" $\\Delta R_{jj}$"),
        Hist.storage.Weight(),
    )
    hists["dphijj"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 4, name="dphijj", label=" $\\Delta \\phi_{jj}$"),
        Hist.storage.Weight(),
    )
    hists["detajj"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 5, name="detajj", label=" $\\Delta \\eta_{jj}$"),
        Hist.storage.Weight(),
    )
    hists["drj0photon"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 5, name="drj0photon", label=" $\\Delta R_{j0\\gamma}$"),
        Hist.storage.Weight(),
    )
    hists["dphij0photon"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 4, name="dphij0photon", label=" $\\Delta \\phi_{j0\\gamma}$"),
        Hist.storage.Weight(),
    )
    hists["detaj0photon"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 5, name="detaj0photon", label=" $\\Delta \\eta_{j0\\gamma}$"),
        Hist.storage.Weight(),
    )
    hists["drj1photon"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 5, name="drj1photon", label=" $\\Delta R_{j1\\gamma}$"),
        Hist.storage.Weight(),
    )
    hists["dphij1photon"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 4, name="dphij1photon", label=" $\\Delta \\phi_{j1\\gamma}$"),
        Hist.storage.Weight(),
    )
    hists["detaj1photon"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 5, name="detaj1photon", label=" $\\Delta \\eta_{j1\\gamma}$"),
        Hist.storage.Weight(),
    )
    hists["ptratiojj"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 3, name="ptratiojj", label=" $p_T^{j0}/p_T^{j1}$"),
        Hist.storage.Weight(),
    )
    hists["ptratioj0photon"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 3, name="ptratioj0photon", label=" $p_T^{j0}/p_T^{\\gamma}$"),
        Hist.storage.Weight(),
    )
    hists["ptratioj1photon"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(50, 0, 3, name="ptratioj1photon", label=" $p_T^{j1}/p_T^{\\gamma}$"),
        Hist.storage.Weight(),
    )

    hists["photonhoe"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(100, 0, 0.1, name="photonhoe", label="Photon H/E"),
        Hist.storage.Weight(),
    )
    hists["photontrkSumPtSolidConeDR04"] = Hist.Hist(
        axes["syst"],
        Hist.axis.Regular(200, 0, 1., name="photontrkSumPtSolidConeDR04", label="Photon trkSumPtSolidConeDR04"),
        Hist.storage.Weight(),
    )

    return hists
