import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
from vardefunc.deband import dumb3kdb
import kagefunc as kgf

# region setup

DEFAULT_PATH = r"[SubsPlease] Tate no Yuusha no Nariagari S2 - 01 (1080p) [515B223B].mkv"
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

sloc = [0, 0.25, 0.5, 1.5, 0.55, 3, 0.6, 6, 0.75, 16, 1, 16]
dft = core.dfttest.DFTTest(src, planes=[0], tbsize=1, slocation=sloc)
knlm = core.knlm.KNLMeansCL(dft, d=2, a=3, h=0.6, channels="UV")

aa = lvf.aa.upscaled_sraa(knlm, 1.5, opencl=True)

deband = dumb3kdb(aa, 16, [24, 16, 16], [16, 4])

grain = kgf.adaptive_grain(deband, 0.2)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

# funny aliasing 10148

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output(
        core.lsmas.LWLibavSource(R"[Judas] Tate no Yuusha no Nariagari - S02E01 INCOMPLETE.mkv")
    )
    debug_output(knlm, "knlm")
    debug_output(aa, "aa")
    debug_output(grain, "grain")
