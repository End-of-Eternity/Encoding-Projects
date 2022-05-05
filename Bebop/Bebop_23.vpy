import subprocess
import sys
import numpy as np
from functools import partial
from pathlib import Path

import vapoursynth as vs
import EoEfunc as eoe
import havsfunc as haf
import lvsfunc as lvf
from debandshit import placebo_deband
from vardefunc.deband import dumb3kdb
from vsutil import get_y, split, join
from vsdpir import DPIR

# region setup

DEFAULT_PATH = r"[JySzE] Cowboy Bebop - 23 [v2].mkv"
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


def choose_tv(clip, tv, not_tv, thr=180, location=(724, 210)):
    assert tv.format == not_tv.format and tv.width == not_tv.width and tv.height == not_tv.height

    if debug:
        tv = core.text.Text(tv, "tv", 9)
        not_tv = core.text.Text(not_tv, "not tv", 9)

    def choose_func(n, f: vs.VideoFrame) -> vs.VideoFrame:
        return tv if np.array(f[0], copy=False)[location[1], location[0]] > thr else not_tv

    fft = core.fftspectrum.FFTSpectrum(get_y(clip), False)
    return core.std.FrameEval(tv, choose_func, fft, [tv, not_tv])


def dfttest(clip, **dfttest_args):
    keys = dfttest_args.keys()
    if not any(param in keys for param in ["slocation", "ssx", "ssy", "sst"]):
        dfttest_args.update(slocation=eoe.freq._slocation)
    if "sst" in keys and ("tbsize" not in keys or keys["tbsize"] < 2):
        raise ValueError("tbsize should be > 1 when using sst")
    kwargs = {**eoe.freq._dfttest_args, **dfttest_args}
    return core.dfttest.DFTTest(clip, **kwargs)


def merge_frequency(low, hi, **dfttest_args):
    hif = core.std.MakeDiff(hi, dfttest(hi, **dfttest_args))
    clip = core.std.MergeDiff(dfttest(low, **dfttest_args), hif)
    return clip


# endregion setup


extract_timestamps(src_path)
src = core.lsmas.LWLibavSource(src_path)
src = core.std.PlaneStats(src)
src8 = src
src = eoe.fmt.set(src, 16)


# region: not tv

cmde_pre = eoe.dn.CMDE(src, 3, 250, 250, prefilter=3, freq_merge=False, contrasharp=True)
bilateral = core.bilateralgpu.Bilateral(cmde_pre, 1, use_shared_memory=False)
bm3d = eoe.dn.BM3D(src, 5, profile="np", ref=bilateral, refine=2)
cmde_post = eoe.dn.CMDE(bm3d, tr=3, thSAD=250, freq_merge=False)
cmde_post = core.std.MaskedMerge(cmde_post, bm3d, core.adg.Mask(src, 24))
cs = eoe.misc.ContraSharpening(cmde_post, cmde_pre)
knlm = core.knlm.KNLMeansCL(cs, d=2, a=3, h=0.7, channels="uv")
aa = lvf.aa.upscaled_sraa(knlm, 1.5, opencl=True)

dehalo = haf.HQDeringmod(
    aa, core.bilateral.Gaussian(aa, 3), mrad=6, msmooth=2, mthr=150, minp=0, darkthr=10, thr=0
)
lmask = core.std.Sobel(get_y(dehalo)).std.Maximum().std.Expr("x 2 *")
dehalo = core.std.MaskedMerge(dehalo, aa, lmask)
dehalo = haf.FineDehalo(dehalo, rx=2, darkstr=0.4, brightstr=1)
dehalo = haf.EdgeCleaner(dehalo, strength=6, rmode=17, smode=1, hot=True)

# fmt: off
deband = core.average.Mean([
    placebo_deband(dehalo, 12, [2, 1], 1, 0),
    placebo_deband(dehalo, 16, [3, 1], 1, 0),
    placebo_deband(dehalo, 20, [3, 1], 1, 0),
    dumb3kdb(dehalo, 16, [28, 16], 0),
])
# fmt: on
deband = placebo_deband(deband, 16, 0, 1, [3, 0])

ntv = deband

# endregion: not tv

# region: tv

OVERLAP = 8

planes = split(src)
# fmt: off
interleaved = core.std.Interleave([
    core.std.Crop(planes[0], 0, src.width // 2 - OVERLAP),
    core.std.Crop(planes[0], src.width // 2 - OVERLAP, 0),
    core.std.AddBorders(core.std.StackVertical(planes[1:]), right=OVERLAP, color=32768)
])
# fmt: on
processed = eoe.fmt.process_as(interleaved, partial(DPIR, strength=10, trt=True), "s")
left, right, chroma = processed[::3], processed[1::3], processed[2::3]
left = core.std.Crop(left, 0, OVERLAP)
right = core.std.Crop(right, OVERLAP, 0)
planes[0] = core.std.StackHorizontal([left, right])
planes[1] = core.std.Crop(chroma, 0, OVERLAP, 0, chroma.height // 2)
planes[2] = core.std.Crop(chroma, 0, OVERLAP, chroma.height // 2, 0)

dpir = join(planes)

tv_cmde = eoe.dn.CMDE(
    dpir,
    3,
    500,
    250,
    refine=5,
    prefilter=core.std.BoxBlur(dpir, 0, 6, 2, 1, 1),
    freq_merge=False,
    contrasharp=False,
)
tv_cmde = merge_frequency(
    dpir,
    tv_cmde,
    slocation=None,
    sigma=1024,
    ssx=[0.0, 0, 0.03, 0, 0.06, 16, 0.15, 56, 0.2, 1024, 1.0, 1024],
    smode=1,
    sbsize=8,
    sosize=6,
)

tv = tv_cmde

# endregion: tv

chosen = choose_tv(src8, tv, ntv)

final = chosen
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(choose_tv(src8, src, src), "src")
    debug_output(src, "src", fft=True)
    debug_output(aa, "aa")
    debug_output(dehalo, "dehalo")
    debug_output(deband, "deband")
    debug_output(tv_cmde, "tv_cmde")
    debug_output(dpir, "dpir")
    debug_output(tv, "tv")
    debug_output(ntv, "ntv")
    debug_output(chosen, "chosen")

print("Eval Done", file=sys.stderr)
