import subprocess
import sys
from pathlib import Path

import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as haf
from debandshit import placebo_deband
from vardefunc.deband import dumb3kdb

# region setup

DEFAULT_PATH = r"[JySzE] Cowboy Bebop - 01 [v2].mkv"
DEFAULT_CACHE = 32
DEFAULT_THREADS = 24

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]


def extract_timestamps(path: str) -> None:
    # workaround to generate only during inital frame count
    if not debug and "max_cache_size" not in globals():
        # delete existing timestamps
        Path("timestamps.txt").exists() and Path("timestamps.txt").unlink()
        print("Extracting timestamps...", file=sys.stderr)
        if subprocess.run(["mkvextract", "timecodes_v2", path, "0:timestamps.txt"]).returncode:
            raise RuntimeError("Failed to extract timestamps")


# endregion setup


extract_timestamps(src_path)
src = core.lsmas.LWLibavSource(src_path)
src = core.std.PlaneStats(src)
src = eoe.fmt.set(src, 16)

# because this script is for a mini, I'm going to be doing far more agressive filtering to get rid
# of the excessive source grain (honestly looks good, but wont compress, at all). This *will* also
# destroy fine details, but they wouldn't have been encoded properly anyway. If you use this script,
# I'd recommend testing lower denoise strengths with some masking.

# A good test scene for this destruction is EP06 frame 29043

# cmde is magic, but the heavy dynamic grain is almost impossible to get rid of without destroying
# lineart in difficult situations. It's also not possible to effectively freq_merge due to the size
# of the grain. cmde also struggles with noise near edges.
cmde_pre = eoe.dn.CMDE(src, 3, 250, 250, prefilter=3, freq_merge=False, contrasharp=True)
# bilateral is used to clean up cmde, and does a somewhat decent job of leaving the lineart alone.
bilateral = core.bilateralgpu.Bilateral(cmde_pre, 1)
# bm3d using cmde+bilateral as a reference. Extremely strong. Manages to repair back many details
# destroyed previously.
bm3d = eoe.dn.BM3D(src, 5, profile="np", ref=bilateral, refine=2)
# bm3d is brings back a little too much of the previous shit, so cmde again to get rid of that.
# more accurate motion vectors generated here seem to do an overall good job.
cmde_post = eoe.dn.CMDE(bm3d, tr=3, thSAD=250, freq_merge=False)
# mask cmde_post using adg, sort of fixes cmde's tendancy to ghost lines in dark areas.
# not as effective as freq_merge, but as above, freq_merge isnt possible.
cmde_post = core.std.MaskedMerge(cmde_post, bm3d, core.adg.Mask(src, 24))
# contra sharp to regain some sharpness.
cs = eoe.misc.ContraSharpening(cmde_post, cmde_pre)

knlm = core.knlm.KNLMeansCL(cs, d=2, a=3, h=0.7, channels="uv")

# reconnect lines. sort of goes against the art design, but so does removing the grain, so nvm.
# some aliasing has been upscaled and is basically impossible to get rid of. see EP01 frame 4142
aa = lvf.aa.upscaled_sraa(knlm, 1.5, opencl=True)

# strong edgecleaner to fix up remaining mosquito noise around lines
dehalo = haf.FineDehalo(aa, rx=2, darkstr=0.3, brightstr=1)
dehalo = haf.EdgeCleaner(dehalo, strength=6, rmode=17, smode=1, hot=True)

deband = core.average.Mean(
    [
        # placebo did a better job on this kind of banding than f3kdb.
        # average to attempt to negate detail loss
        placebo_deband(dehalo, 12, [2, 1], 1, 0),
        placebo_deband(dehalo, 16, [3, 1], 1, 0),
        placebo_deband(dehalo, 20, [3, 1], 1, 0),
        dumb3kdb(dehalo, 16, [28, 16], 0),
    ]
)
# no debanding, only grain. adaptive_grain unnecessary after this
deband = placebo_deband(deband, 16, 0, 1, [3, 0])

final = deband
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output("[Judas] Cowboy Bebop - S01E01 - Script v0.mkv", "encode")
    debug_output(cmde_pre, "cmde")
    debug_output(bilateral, "bilateral")
    debug_output(bm3d, "bm3d")
    debug_output(knlm, "knlm")
    debug_output(aa, "aa")
    debug_output(dehalo, "dehalo")
    debug_output(deband, "deband")
