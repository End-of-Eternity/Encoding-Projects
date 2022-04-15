import re

import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import kagefunc as kgf
from vardefunc.deband import dumb3kdb
from vsutil import get_y


# region setup

DEFAULT_PATH = r"Date A Live - S04E01 - 1080p WEB x264 -NanDesuKa (CR).mkv"
DEFAULT_CACHE = 32
DEFAULT_THREADS = 24

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]

# endregion setup

ep = re.match(r"Date A Live - S04E(\d\d) - 1080p WEB x264 -NanDesuKa \(CR\)\.mkv", src_path)

if ep is None:
    raise ValueError("Couldn't find episode number in file name")
ep = ep.group(1)

cr = core.lsmas.LWLibavSource(src_path)
funi = core.lsmas.LWLibavSource(
    rf"src/Date A Live - S04E{ep} - 1080p WEB H.264 -NanDesuKa (FUNi).mkv"
)
funi = funi[240:]
cr = cr[: funi.num_frames]

cmde_cr = eoe.dn.CMDE(cr, 3, 100, refine=2)
cmde_funi = eoe.dn.CMDE(get_y(funi), 3, 100, refine=2)

# weight CR a little higher because it's better quality
cmde = core.std.Expr([get_y(cmde_cr), cmde_funi], "x 6 * y 5 * + 11 /")
cmde = core.std.ShufflePlanes([cmde, cmde_cr], [0, 1, 2], vs.YUV)

sloc = [0, 0, 0.4, 1, 0.45, 3, 0.5, 5, 1, 32]
dft = core.dfttest.DFTTest(cmde, tbsize=1, slocation=sloc)
bm3d = eoe.dn.BM3D(dft, [1.3, 0], 1, "lc")
knlm = core.knlm.KNLMeansCL(bm3d, d=2, a=3, h=0.6, channels="UV")

aa = lvf.aa.nnedi3(knlm, opencl=True)

# fmt: off
deband = core.average.Mean([
    dumb3kdb(aa, 8, [36, 24, 24], 0),
    dumb3kdb(aa, 16, [36, 24, 24], 0),
])
deband = dumb3kdb(deband, 16, 1, [16, 4])
# fmt: on

grain = kgf.adaptive_grain(deband, 0.2)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(cr, "cr")
    debug_output(funi, "funi")
    debug_output(cmde, "cmde")
    debug_output(knlm, "knlm")
    debug_output(aa, "aa")
    debug_output(deband, "deband")
    debug_output(grain, "grain")
