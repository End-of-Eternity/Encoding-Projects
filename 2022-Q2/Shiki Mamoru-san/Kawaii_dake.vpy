import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as haf
import kagefunc as kgf
from vardefunc.deband import dumb3kdb

# region setup

DEFAULT_PATH = r"Kawaii dake ja Nai Shikimori-san - S01E01 - 1080p WEB x264 -NanDesuKa (CR).mkv"
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

bm3d = eoe.dn.BM3D(src, [1.2, 0], 1, "lc")
knlm = core.knlm.KNLMeansCL(bm3d, d=2, a=3, h=0.6, channels="UV")

dehalo = haf.FineDehalo(knlm, rx=3, darkstr=0, brightstr=1)
dehalo = haf.EdgeCleaner(dehalo, strength=2, rmode=13, smode=1, hot=True)

deband = dumb3kdb(dehalo, 16, [30, 24, 24], [16, 4])

grain = kgf.adaptive_grain(deband, 0.25)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output(knlm, "knlm")
    debug_output(dehalo, "dehalo")
    debug_output(grain, "grain")
