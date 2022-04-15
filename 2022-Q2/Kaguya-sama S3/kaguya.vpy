import re

import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as haf
import kagefunc as kgf
from vardefunc.deband import dumb3kdb
from vsutil import get_y, plane


# region setup

DEFAULT_PATH = (
    r"Kaguya-sama wa Kokurasetai - Tensai-tachi no Renai Zunousen - S03E02 - 1080p WEB x264"
    r" -NanDesuKa (CR).mkv"
)
DEFAULT_CACHE = 32
DEFAULT_THREADS = 24

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]

# endregion setup

ep = re.match(
    r"Kaguya-sama wa Kokurasetai - Tensai-tachi no Renai Zunousen - S03E(\d\d) - 1080p WEB x264"
    r" -NanDesuKa \(CR\)\.mkv",
    src_path,
)

if ep is None:
    raise ValueError("Couldn't find episode number in file name")
ep = ep.group(1)

b_avc = f"src/Kaguya-sama wa Kokurasetai S3 - {ep} - 1080p WEB H.264 -NanDesuKa (B-Global).mkv"
b_hevc = f"src/Kaguya-sama wa Kokurasetai S3 - {ep} - 1080p WEB HEVC -NanDesuKa (B-Global).mkv"

cr = core.lsmas.LWLibavSource(src_path)
b_avc = core.lsmas.LWLibavSource(b_avc)
b_hevc = core.lsmas.LWLibavSource(b_hevc)

cr = eoe.fmt.set(cr, "16").std.AssumeFPS(fpsnum=24000, fpsden=1001)
b_avc = eoe.fmt.set(b_avc, "16")
b_hevc = eoe.fmt.set(b_hevc, "16")

cmde_b_hevc = eoe.dn.CMDE(get_y(b_hevc), 3, 100, refine=1, globalmode=eoe.dn.GlobalMode.WRITE)
cmde_cr = eoe.dn.CMDE(get_y(cr), 3, 100, globalmode=eoe.dn.GlobalMode.READ)
cmde_b_avc = eoe.dn.CMDE(get_y(b_avc), 3, 100, globalmode=eoe.dn.GlobalMode.READ)

cmde_luma = core.std.Expr([cmde_cr, cmde_b_avc, cmde_b_hevc], "x 6 * y 8 * z 9 * + + 23 /")
# minorly better performance than just averaging everything (~1.5%)
chroma_u = core.average.Mean([plane(cr, 1), plane(b_avc, 1), plane(b_hevc, 1)])
chroma_v = core.average.Mean([plane(cr, 2), plane(b_avc, 2), plane(b_hevc, 2)])
cmde = core.std.ShufflePlanes([cmde_luma, chroma_u, chroma_v], planes=[0, 0, 0], colorfamily=vs.YUV)

rs = eoe.rs.rescale(
    cmde,
    1554,
    874,
    "bicubic",
    b=0,
    c=0.5,
    mask_detail=True,
    rescale_threshold=0.05,
    morpho=False,
    maximum=4,
    inflate=2,
)

bm3d = eoe.dn.BM3D(rs, [1.2, 1.6], 1, "lc")

aa = lvf.aa.nnedi3(bm3d, opencl=True)

dehalo = haf.FineDehalo(aa, rx=2.4, darkstr=0, brightstr=1, contra=1)
dehalo = haf.EdgeCleaner(dehalo, strength=4, rmode=13, smode=1, hot=True)

deband = dumb3kdb(dehalo, 16, [32, 24, 24], [16, 4])

grain = kgf.adaptive_grain(deband, 0.2, luma_scaling=4)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(b_hevc, "b_hevc")
    debug_output(b_avc, "b_avc")
    debug_output(cr, "cr")
    debug_output(cmde, "cmde")
    debug_output(rs, "rs")
    debug_output(bm3d, "bm3d")
    debug_output(dehalo, "dehalo")
    debug_output(deband, "deband")
