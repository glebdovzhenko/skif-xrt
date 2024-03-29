ring_kwargs = {
    "eE": 3.0,  # electron energy [GeV]
    "eEspread": 0.00135,  # electron energy spread, relative to eE, [root-mean-square]
    "eI": 0.4,  # ring current, [A]
    "betaX": 15.66,  # Betatron function X [m]
    "betaZ": 2.29,  # Betatron function Z [m]
    "eEpsilonX": 9.586e-2,  # Electron beam emittance X [nm * rad]
    "eEpsilonZ": 9.586e-3,  # Electron beam emittance Z [nm * rad]
}

wiggler_1_3_kwargs = {
    "K": 6.80688,  # deflection parameter
    "period": 27.0,  # period length [mm]
    "n": 74,  # number of periods
}

wiggler_1_5_kwargs = {
    "K": 20.1685,  # deflection parameter
    "period": 48.0,  # period length [mm]
    "n": 18,  # number of periods
}

undulator_1_1_kwargs = {
    "K": 1.82,  # deflection parameter
    "period": 15.6,  # period length [mm]
    "n": 128,  # number of periods
}

wiggler_nstu_scw_kwargs = {
    "K": 13.8939,  # deflection parameter at 3.1 T, at 4.5 T K = 20.2
    "period": 48.0,  # period length [mm]
    "n": 40,  # number of periods
}
