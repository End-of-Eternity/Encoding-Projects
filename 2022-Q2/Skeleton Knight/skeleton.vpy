import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import kagefunc as kgf
import havsfunc as haf
from vardefunc.deband import dumb3kdb
from vsutil import get_y, iterate

# region setup

DEFAULT_PATH = (
    r"[Erai-raws] Gaikotsu Kishi-sama Tadaima Isekai e Odekake-chuu - 01 [1080p][Multiple"
    r" Subtitle][1D9B823B].mkv"
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

sloc = [0, 0.25, 0.5, 1.5, 0.55, 3, 0.6, 6, 0.75, 16, 1, 16]
dft = core.dfttest.DFTTest(src, planes=[0], tbsize=1, slocation=sloc)
knlm = core.knlm.KNLMeansCL(dft, d=2, a=3, h=0.6, channels="UV")

credit_mask = core.std.Expr(get_y(knlm), f"x {0.85*65536} > 65535 0 ?")
credit_mask = core.median.TemporalMedian(credit_mask, 12)
credit_mask = iterate(credit_mask, core.std.Maximum, 4)
credit_mask = iterate(credit_mask, core.std.Inflate, 2)

aa_strong = lvf.aa.upscaled_sraa(knlm, 1.3, opencl=True)
aa_weak = lvf.aa.upscaled_sraa(knlm, 1.6, opencl=True)
aa_credits = lvf.aa.clamp_aa(knlm, aa_strong, aa_weak)
aa_credits = core.std.MaskedMerge(aa_credits, knlm, credit_mask)
aa = aa_weak[:-2400] + aa_credits[-2400:]

dehalo = haf.FineDehalo(aa, darkstr=0, brightstr=0.8)

deband = dumb3kdb(dehalo, 16, [24, 16, 16], [16, 4])

grain = kgf.adaptive_grain(deband, 0.2)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output(knlm, "knlm")
    debug_output(aa, "aa")
    debug_output(credit_mask, "credit_mask")
    debug_output(grain, "grain")
