# MF6_to_FloPy
Translate an existing **MODFLOW 6 model folder** (mfsim.nam + package files) into a clean, readable **FloPy build script** of explicit constructor calls, e.g. `ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)
