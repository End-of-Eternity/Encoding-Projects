import re

import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import kagefunc as kgf
from vsTAAmbk import TAAmbk as taa
from vsutil import get_y, split, join
from vardefunc.deband import dumb3kdb

# region setup

DEFAULT_PATH = r"Yuusha, Yamemasu - 01 - 1080p WEB HEVC -NanDesuKa (B-Global).mkv"
DEFAULT_CACHE = 32
DEFAULT_THREADS = 24

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]

# endregion setup

ep = re.match(r"Yuusha, Yamemasu - (\d\d) - 1080p WEB HEVC -NanDesuKa \(B-Global\)\.mkv", src_path)

if ep is None:
    raise ValueError("Couldn't find episode number in file name")

src_hevc = core.lsmas.LWLibavSource(src_path)
src_avc = core.lsmas.LWLibavSource(
    f"src/Yuusha, Yamemasu - {ep.group(1)} - 1080p WEB H.264 -NanDesuKa (B-Global).mkv"
)
src_hevc = eoe.fmt.set(src_hevc, "16")
src_avc = eoe.fmt.set(src_avc, "16")

src = core.std.ShufflePlanes(
    [core.average.Mean([get_y(src_hevc), get_y(src_avc)]), src_hevc], [0, 1, 2], vs.YUV
)

cmde = eoe.dn.CMDE(src, tr=2, thSAD=100)
dark_mask = core.adg.Mask(cmde.std.PlaneStats(), 24)
cmde = core.std.MaskedMerge(cmde, src, dark_mask)

sloc = [0, 0.25, 0.5, 1, 0.55, 2, 0.6, 4, 0.75, 16, 1, 16]
dft = core.dfttest.DFTTest(cmde, planes=[0], tbsize=1, slocation=sloc)
bm3d = eoe.dn.BM3D(dft, [1, 2], 0, "np")

planes = split(bm3d)
# can't believe I'm unironically using SangNom
planes[0] = lvf.aa.clamp_aa(
    planes[0], lvf.aa.nnedi3(planes[0], opencl=True), taa(planes[0], -3, opencl=True)
)
# Use this if you don't care about performance and want better quality than SangNom
# planes[0] = lvf.aa.nneedi3_clamp(planes[0], opencl=True)
planes[1] = lvf.aa.nnedi3(planes[1], opencl=True)
planes[2] = lvf.aa.nnedi3(planes[2], opencl=True)
aa = join(planes)

deband = dumb3kdb(aa, 16, [30, 24, 24], [16, 4])

grain = kgf.adaptive_grain(deband, 0.2)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output(cmde, "cmde")
    debug_output(dark_mask, "dark_mask")
    debug_output(bm3d, "bm3d")
    debug_output(aa, "aa")
    debug_output(grain, "grain")
