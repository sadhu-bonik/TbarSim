"""
Full validation of Tbar (crossbar.py) against an Xyce SPICE reference run,
for a 3x3x3 memristor crossbar array.

CONFIRMED PARAMETERS (derived via KCL cross-checks against the Xyce data,
not guessed -- see the accompanying chat for the derivation):
    - The 10 / 1000 values in the netlist are RESISTANCES (ohms), not
      conductances. Convert with G = 1/R -> 0.1 S / 0.001 S.
    - S (input conductance)  = 1,000,000
    - L (output conductance) = 1,000,000
    - GW (wire/via conductance) = 1000
    - RS_ik is the INPUT resistor: connects the 3.2V source to W(i,0,k)
    - RL_jk is the OUTPUT resistor: connects B(n-1,j,k) to O(j,k) at 0V

NOTE ON CURRENT SIGNS: Tbar.get_current() returns a magnitude (no direction),
while Xyce reports signed, direction-dependent currents. This script compares
magnitudes for currents, which is the correct comparison against our simulator.

Run this file directly:  python3 validate_full.py
"""

import re
from crossbar import Tbar

# ---------------------------------------------------------------------------
# 1. Xyce reference data (header row + single data row, exactly as reported)
# ---------------------------------------------------------------------------
_HEADER = """Index V(NODE_W000) V(NODE_W001) V(NODE_W002) V(NODE_W010) V(NODE_W011) V(NODE_W012) V(NODE_W020) V(NODE_W021) V(NODE_W022) V(NODE_W100) V(NODE_W101) V(NODE_W102) V(NODE_W110) V(NODE_W111) V(NODE_W112) V(NODE_W120) V(NODE_W121) V(NODE_W122) V(NODE_W200) V(NODE_W201) V(NODE_W202) V(NODE_W210) V(NODE_W211) V(NODE_W212) V(NODE_W220) V(NODE_W221) V(NODE_W222) V(NODE_B000) V(NODE_B001) V(NODE_B002) V(NODE_B010) V(NODE_B011) V(NODE_B012) V(NODE_B020) V(NODE_B021) V(NODE_B022) V(NODE_B100) V(NODE_B101) V(NODE_B102) V(NODE_B110) V(NODE_B111) V(NODE_B112) V(NODE_B120) V(NODE_B121) V(NODE_B122) V(NODE_B200) V(NODE_B201) V(NODE_B202) V(NODE_B210) V(NODE_B211) V(NODE_B212) V(NODE_B220) V(NODE_B221) V(NODE_B222) I(RS_00) I(RS_10) I(RS_20) I(RS_01) I(RS_11) I(RS_21) I(RS_02) I(RS_12) I(RS_22) I(RL_00) I(RL_10) I(RL_20) I(RL_01) I(RL_11) I(RL_21) I(RL_02) I(RL_12) I(RL_22) I(RWX_000) I(RWX_010) I(RWX_001) I(RWX_011) I(RWX_002) I(RWX_012) I(RWX_100) I(RWX_110) I(RWX_101) I(RWX_111) I(RWX_102) I(RWX_112) I(RWX_200) I(RWX_210) I(RWX_201) I(RWX_211) I(RWX_202) I(RWX_212) I(RWY_200) I(RWY_201) I(RWY_202) I(RWY_210) I(RWY_211) I(RWY_212) I(RWY_220) I(RWY_221) I(RWY_222) I(RWY_100) I(RWY_101) I(RWY_102) I(RWY_110) I(RWY_111) I(RWY_112) I(RWY_120) I(RWY_121) I(RWY_122) I(RWZW_000) I(RWZW_001) I(RWZW_010) I(RWZW_011) I(RWZW_020) I(RWZW_021) I(RWZW_100) I(RWZW_101) I(RWZW_110) I(RWZW_111) I(RWZW_120) I(RWZW_121) I(RWZW_200) I(RWZW_201) I(RWZW_210) I(RWZW_211) I(RWZW_220) I(RWZW_221) I(RWZB_000) I(RWZB_001) I(RWZB_010) I(RWZB_011) I(RWZB_020) I(RWZB_021) I(RWZB_100) I(RWZB_101) I(RWZB_110) I(RWZB_111) I(RWZB_120) I(RWZB_121) I(RWZB_200) I(RWZB_201) I(RWZB_210) I(RWZB_211) I(RWZB_220) I(RWZB_221) I(RG_000) I(RG_001) I(RG_002) I(RG_010) I(RG_011) I(RG_012) I(RG_020) I(RG_021) I(RG_022) I(RG_100) I(RG_101) I(RG_102) I(RG_110) I(RG_111) I(RG_112) I(RG_120) I(RG_121) I(RG_122) I(RG_200) I(RG_201) I(RG_202) I(RG_210) I(RG_211) I(RG_212) I(RG_220) I(RG_221) I(RG_222)"""

_VALUES = """0 3.19999937e+00 3.19999964e+00 3.19999937e+00 3.19969302e+00 3.19964306e+00 3.19969302e+00 3.19943981e+00 3.19950650e+00 3.19943981e+00 3.19999966e+00 3.19999939e+00 3.19999966e+00 3.19965979e+00 3.19970975e+00 3.19965979e+00 3.19958990e+00 3.19952321e+00 3.19958990e+00 3.19999937e+00 3.19999964e+00 3.19999937e+00 3.19969297e+00 3.19964301e+00 3.19969297e+00 3.19943972e+00 3.19950641e+00 3.19943972e+00 5.60282295e-04 4.93586335e-04 5.60282295e-04 4.10099647e-04 4.76790170e-04 4.10099647e-04 5.60187099e-04 4.93503250e-04 5.60187099e-04 3.07034346e-04 3.56994908e-04 3.07034346e-04 3.40209842e-04 2.90254588e-04 3.40209842e-04 3.06982985e-04 3.56936539e-04 3.06982985e-04 6.26142302e-07 3.60366094e-07 6.26142302e-07 3.43331949e-07 6.09078256e-07 3.43331949e-07 6.26035131e-07 3.60307193e-07 6.26035131e-07 6.26035131e-01 3.43331949e-01 6.26142302e-01 3.60307193e-01 6.09078256e-01 3.60366094e-01 6.26035131e-01 3.43331949e-01 6.26142302e-01 6.26142302e-01 3.43331949e-01 6.26035131e-01 3.60366094e-01 6.09078256e-01 3.60307193e-01 6.26142302e-01 3.43331949e-01 6.26035131e-01 3.06356950e-01 2.53204114e-01 3.56576231e-01 1.36566711e-01 3.06356950e-01 2.53204114e-01 3.39866510e-01 6.98898055e-02 2.89645509e-01 1.86535582e-01 3.39866510e-01 6.98898055e-02 3.06408204e-01 2.53247949e-01 3.56634542e-01 1.36591427e-01 3.06408204e-01 2.53247949e-01 -3.06408204e-01 -3.56634542e-01 -3.06408204e-01 -3.39866510e-01 -2.89645509e-01 -3.39866510e-01 -3.06356950e-01 -3.56576231e-01 -3.06356950e-01 -2.53247949e-01 -1.36591427e-01 -2.53247949e-01 -6.98898055e-02 -1.86535582e-01 -6.98898055e-02 -2.53204114e-01 -1.36566711e-01 -2.53204114e-01 -2.65727938e-04 2.65727938e-04 4.99535535e-02 -4.99535535e-02 -6.66838490e-02 6.66838490e-02 2.65746308e-04 -2.65746308e-04 -4.99552539e-02 4.99552539e-02 6.66905226e-02 -6.66905226e-02 -2.65776208e-04 2.65776208e-04 4.99605621e-02 -4.99605621e-02 -6.66959603e-02 6.66959603e-02 6.66959603e-02 -6.66959603e-02 -6.66905226e-02 6.66905226e-02 6.66838490e-02 -6.66838490e-02 -4.99605621e-02 4.99605621e-02 4.99552539e-02 -4.99552539e-02 -4.99535535e-02 4.99535535e-02 2.65776209e-04 -2.65776209e-04 -2.65746308e-04 2.65746308e-04 2.65727938e-04 -2.65727938e-04 3.19943909e-01 3.19950605e-03 3.19943909e-01 3.19928292e-03 3.19916627e-01 3.19928292e-03 3.19887963e-01 3.19901299e-03 3.19887963e-01 3.19969262e-03 3.19964240e-01 3.19969262e-03 3.19931958e-01 3.19941949e-03 3.19931958e-01 3.19928292e-03 3.19916627e-01 3.19928292e-03 3.19999875e-01 3.19999928e-03 3.19999875e-01 3.19969262e-03 3.19964240e-01 3.19969262e-03 3.19943909e-01 3.19950605e-03 3.19943909e-01"""


def load_xyce_data():
    headers = _HEADER.split()[1:]           # drop "Index" label
    values = [float(x) for x in _VALUES.split()[1:]]  # drop the "0" row index
    assert len(headers) == len(values), (len(headers), len(values))
    return dict(zip(headers, values))


# ---------------------------------------------------------------------------
# 2. Build and solve our simulator with the confirmed parameters
# ---------------------------------------------------------------------------
def build_reference_model():
    cb = Tbar(n=3, m=3, p=3, S=1e6, L=1e6, GW=1000)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                R = 10 if (i + j + k) % 2 == 0 else 1000   # netlist value is OHMS
                cb.set_conductance(i, j, k, 1.0 / R)
    cb.set_bias(vin=3.2, vout=0.0)
    cb.solve()
    return cb


# ---------------------------------------------------------------------------
# 3. Compare every value; currents compared by magnitude (see note at top)
# ---------------------------------------------------------------------------
def rel_err(ours, ref):
    ref_mag = abs(ref)
    if ref_mag == 0:
        return abs(ours)  # absolute error when reference is exactly zero
    return abs(abs(ours) - ref_mag) / ref_mag * 100


def compare_all(cb, data):
    results = []  # (label, ours, xyce, pct_err)

    for key, val in data.items():
        m = re.match(r"V\(NODE_([WB])(\d)(\d)(\d)\)", key)
        if m:
            node_type, i, j, k = m.group(1), *map(int, m.groups()[1:])
            ours = cb.get_voltage(f"{node_type}({i},{j},{k})")
            results.append((key, ours, val, rel_err(ours, val)))
            continue

        m = re.match(r"I\(RG_(\d)(\d)(\d)\)", key)
        if m:
            i, j, k = map(int, m.groups())
            ours = cb.get_current(f"W({i},{j},{k})", f"B({i},{j},{k})")
            results.append((key, ours, val, rel_err(ours, val)))
            continue

        m = re.match(r"I\(RS_(\d)(\d)\)", key)  # input resistor
        if m:
            i, k = map(int, m.groups())
            ours = cb.get_current(f"I({i},{k})", f"W({i},0,{k})")
            results.append((key, ours, val, rel_err(ours, val)))
            continue

        m = re.match(r"I\(RL_(\d)(\d)\)", key)  # output resistor
        if m:
            j, k = map(int, m.groups())
            ours = cb.get_current(f"B(2,{j},{k})", f"O({j},{k})")
            results.append((key, ours, val, rel_err(ours, val)))
            continue

        m = re.match(r"I\(RWX_(\d)(\d)(\d)\)", key)  # wordline segment
        if m:
            i, j, k = map(int, m.groups())
            ours = cb.get_current(f"W({i},{j},{k})", f"W({i},{j+1},{k})")
            results.append((key, ours, val, rel_err(ours, val)))
            continue

        m = re.match(r"I\(RWY_(\d)(\d)(\d)\)", key)  # bitline segment
        if m:
            i, j, k = map(int, m.groups())
            ours = cb.get_current(f"B({i-1},{j},{k})", f"B({i},{j},{k})")
            results.append((key, ours, val, rel_err(ours, val)))
            continue

        m = re.match(r"I\(RWZW_(\d)(\d)(\d)\)", key)  # vertical via, wordline
        if m:
            i, j, k = map(int, m.groups())
            ours = cb.get_current(f"W({i},{j},{k})", f"W({i},{j},{k+1})")
            results.append((key, ours, val, rel_err(ours, val)))
            continue

        m = re.match(r"I\(RWZB_(\d)(\d)(\d)\)", key)  # vertical via, bitline
        if m:
            i, j, k = map(int, m.groups())
            ours = cb.get_current(f"B({i},{j},{k})", f"B({i},{j},{k+1})")
            results.append((key, ours, val, rel_err(ours, val)))
            continue

    return results


def main():
    data = load_xyce_data()
    cb = build_reference_model()
    results = compare_all(cb, data)

    errors = [r[3] for r in results]
    print(f"Total values compared: {len(results)}")
    print(f"Average % error: {sum(errors) / len(errors):.6f}%")
    print(f"Median  % error: {sorted(errors)[len(errors)//2]:.6f}%")
    print(f"Max     % error: {max(errors):.6f}%")
    print()

    worst = sorted(results, key=lambda r: -r[3])[:10]
    print("Worst 10 matches:")
    for label, ours, xyce, err in worst:
        print(f"  {label:16s} ours={ours: .8e}  xyce={xyce: .8e}  err={err:.6f}%")

    print()
    print("Full per-value breakdown:")
    for label, ours, xyce, err in results:
        print(f"  {label:16s} ours={ours: .8e}  xyce={xyce: .8e}  err={err:.6f}%")


if __name__ == "__main__":
    main()