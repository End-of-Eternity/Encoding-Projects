import re

import vapoursynth as vs
import EoEfunc as eoe
import havsfunc as haf
import kagefunc as kgf
from vardefunc.deband import dumb3kdb


# region setup

DEFAULT_PATH = r"Mahoutsukai Reimeiki - 01 - 1080p WEB x264 -NanDesuKa (CR).mkv"
DEFAULT_CACHE = 32
DEFAULT_THREADS = 24

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]

# endregion setup

ep = re.match(r"Mahoutsukai Reimeiki - (\d\d) - 1080p WEB x264 -NanDesuKa \(CR\)\.mkv", src_path)

if ep is None:
    raise ValueError("Could not get episode from filename")
ep = ep.group(1)

src_cr = core.lsmas.LWLibavSource(src_path)

src_1080_hevc = f"src/Mahoutsukai Reimeiki - {ep} - 1080p WEB HEVC -NanDesuKa (B-Global).mkv"
src_2160_hevc = f"src/Mahoutsukai Reimeiki - {ep} - 2160p WEB HEVC -NanDesuKa (B-Global).mkv"
src_2160_avc = f"src/Mahoutsukai Reimeiki - {ep} - 2160p WEB H.264 -NanDesuKa (B-Global).mkv"
src_1080_hevc = core.lsmas.LWLibavSource(src_1080_hevc)
src_2160_hevc = core.lsmas.LWLibavSource(src_2160_hevc)
src_2160_avc = core.lsmas.LWLibavSource(src_2160_avc)

resize_args = dict(filter_param_a=0, filter_param_b=0.5, format=vs.YUV420P16)
src_1080_hevc = eoe.fmt.set(src_1080_hevc, "16")
src_2160_hevc = core.resize.Bicubic(src_2160_hevc, 1920, 1080, **resize_args)
src_2160_avc = core.resize.Bicubic(src_2160_avc, 1920, 1080, **resize_args)

cmde_1080_hevc = eoe.dn.CMDE(src_1080_hevc, 3, 150, refine=1)
cmde_2160_hevc = eoe.dn.CMDE(src_2160_hevc, 3, 150, refine=1)
cmde_2160_avc = eoe.dn.CMDE(src_2160_avc, 3, 150, refine=1)

cmde = core.average.Mean([cmde_1080_hevc, cmde_2160_avc, cmde_2160_hevc])

sloc = [0, 0.25, 0.5, 1.5, 0.5, 4, 0.6, 8, 0.75, 16, 1, 16]
dft = core.dfttest.DFTTest(cmde, planes=[0], tbsize=1, slocation=sloc)
bm3d = eoe.dn.BM3D(dft, [1, 0], 1, "lc")
knlm = core.knlm.KNLMeansCL(bm3d, d=2, a=3, h=0.6, channels="UV")

dehalo = haf.FineDehalo(knlm, darkstr=0, brightstr=0.7)

deband = dumb3kdb(dehalo, 16, [24, 24, 24], [16, 4])

grain = kgf.adaptive_grain(deband, 0.25)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src_1080_hevc, "src_1080_hevc")
    debug_output(src_cr, "src_sp")
    debug_output(
        core.lsmas.LWLibavSource(
            "[Judas] Mahoutsukai Reimeiki (The Dawn of the Witch) - S01E01.mkv"
        ),
        "encode",
    )
    # debug_output(src_2160_hevc, "src_2160_hevc")
    # debug_output(src_2160_avc, "src_2160_avc")
    debug_output(cmde, "cmde")
    debug_output(knlm, "knlm")
    debug_output(dehalo, "dehalo")
    debug_output(grain, "grain")
