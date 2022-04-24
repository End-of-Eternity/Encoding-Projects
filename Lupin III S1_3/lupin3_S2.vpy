from vardefunc.deband import dumb3kdb
import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as haf

# region setup

DEFAULT_PATH = r"00001.m2ts"
DEFAULT_CACHE = 32
DEFAULT_THREADS = 24

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]

# endregion setup

src = core.lsmas.LWLibavSource(src_path)
src = eoe.fmt.set(src, "16")

# crappy edges moment
src = core.std.Crop(src, 244, 244, 4, 4)

# horrible chroma moment
cmde = eoe.dn.CMDE(src, thSAD=100, thSADC=200)
bm3d = eoe.dn.BM3D(cmde, [1.1], 0)
knlm = core.knlm.KNLMeansCL(bm3d, d=2, a=3, h=1.0)

# crap noise means moving this later is a good idea
cs = eoe.misc.ContraSharpening(knlm, src)

ef = core.edgefixer.ContinuityFixer(cs, 1, 1, 1, 2)

aa = lvf.aa.upscaled_sraa(cs, 2)

dehalo = haf.FineDehalo(aa, darkstr=0.1, brightstr=0.8)

cwarp = core.warp.AWarpSharp2(dehalo, blur=2, type=1, depth=4, planes=[1, 2])

slocation = [0, 0, 0.5, 0, 0.6, 1, 0.7, 1.5, 0.8, 4, 0.9, 6, 1, 16]
dechecker = core.dfttest.DFTTest(cwarp, sigma=2, slocation=slocation, **eoe.freq._dfttest_args)

deband = core.average.Mean(
    [
        dumb3kdb(cwarp, 14, [24, 32, 32]),
        dumb3kdb(cwarp, 18, [30, 38, 38]),
    ]
)
deband = dumb3kdb(deband, threshold=1, grain=[16, 6])

grain = eoe.misc.sized_grain(deband, [0.3, 0.1], 6, False)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output(cs, "denoise")
    debug_output(aa, "aa")
    debug_output(cwarp, "cwarp")
    debug_output(dechecker, "dechecker")
    debug_output(deband, "deband")
    debug_output(grain, "grain")
