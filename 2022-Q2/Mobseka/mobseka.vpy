import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
from vardefunc.deband import dumb3kdb
import kagefunc as kgf

# region setup

DEFAULT_PATH = (
    r"[SubsPlease] Otome Game Sekai wa Mob ni Kibishii Sekai desu - 01 (1080p) [D7F51270].mkv"
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

cmde = eoe.dn.CMDE(src, 2, 100)

bm3d = eoe.dn.BM3D(cmde, [1, 0], 0, "fast")
knlm = core.knlm.KNLMeansCL(bm3d, d=2, a=3, h=0.9, channels="UV")

aa = lvf.aa.clamp_aa(
    knlm, lvf.aa.nnedi3(knlm), lvf.aa.upscaled_sraa(knlm, 1.3, 13, eedi3cl=True), 4
)

deband = dumb3kdb(aa, 16, 32, [8, 2])

grain = kgf.adaptive_grain(deband, 0.25)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    # debug_output(src, "src", fft=True)
    debug_output(cmde, "cmde")
    debug_output(bm3d, "dft")
    debug_output(knlm, "knlm")
    debug_output(aa, "aa")
    debug_output(grain, "grain")
