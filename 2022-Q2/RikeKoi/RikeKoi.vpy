import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as haf
import kagefunc as kgf
from vardefunc.deband import dumb3kdb

# region setup

DEFAULT_PATH = (
    r"[SubsPlease] Rikei ga Koi ni Ochita no de Shoumei shitemita S2 - 01 (1080p) [6DB3465C].mkv"
)
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

cmde = eoe.dn.CMDE(src, thSAD=100, tr=3)

aa = lvf.aa.upscaled_sraa(cmde, 1.6, opencl=True)

dft = core.dfttest.DFTTest(
    aa, tbsize=1, planes=[0], slocation=[0, 0, 0.45, 0, 0.5, 2, 0.55, 4, 0.75, 8, 1, 16]
)
knlm = core.knlm.KNLMeansCL(dft, d=2, a=3, h=0.7)

cs = eoe.misc.ContraSharpening(knlm, aa.rgvs.Repair(cmde, 14))

dehalo = haf.FineDehalo(cs, darkstr=0.0, brightstr=0.8)

cwarp = core.warp.AWarpSharp2(dehalo, blur=2, type=1, depth=2, planes=[1, 2])

deband = dumb3kdb(cwarp, 16, [24, 30, 30], [16, 4])

grain = kgf.adaptive_grain(deband, 0.2)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output(cmde, "cmde")
    debug_output(aa, "aa")
    debug_output(cs, "cs")
    debug_output(dehalo, "dehalo")
    debug_output(cwarp, "cwarp")
    debug_output(grain, "grain")
    # debug_output(src, "src", fft=True)
