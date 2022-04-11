import vapoursynth as vs
import EoEfunc as eoe

# region setup

DEFAULT_PATH = r"01_incomplete.mkv"
DEFAULT_CACHE = 32
DEFAULT_THREADS = 24

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]

# endregion setup

src = core.lsmas.LWLibavSource(src_path)
comp = core.lsmas.LWLibavSource(
    R"../Yuusha, Yamemasu - 01 - 1080p WEB x264 -NanDesuKa (HIDIVE).mkv"
)[24:]


final = src
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output(comp, "comp")
