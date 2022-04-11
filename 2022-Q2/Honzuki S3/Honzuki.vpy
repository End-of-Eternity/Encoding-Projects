import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as haf
import kagefunc as kgf
from vardefunc.deband import dumb3kdb

# region setup

DEFAULT_PATH = r"Honzuki no Gekokujou - S03E01 - 1080p WEB x264 -NanDesuKa (CR).mkv"
DEFAULT_CACHE = 32
DEFAULT_THREADS = 24

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]


def rescale(
    src: vs.VideoNode,
    width: int = None,
    height: int = 0,
    kernel: str = "bilinear",
    taps: int = 4,
    b: float = 1 / 3,
    c: float = 1 / 3,
    mask_detail: bool = False,
    rescale_threshold: float = 0.015,
    get_mask: bool = False,
    **mask_args,
) -> vs.VideoNode:
    from EoEfunc.format import set_format, make_similar, make_similar_mask
    from vsutil import split, join, get_y
    import kagefunc as kgf
    import fvsfunc as fvf
    import nnedi3_rpow2

    if not height:
        raise ValueError("rescale: you need to specify a value for the output height")

    only_luma = src.format.num_planes == 1

    planes = split(src)
    planes[0] = descaled = make_similar(
        kgf._descale_luma(set_format(planes[0], "s"), width, height, kernel, taps, b, c), planes[0]
    )

    # denoise whilst descaled
    planes[0] = eoe.dn.BM3D(planes[0], [1.1, 0], 1, "lc")

    planes[0] = nnedi3_rpow2.nnedi3cl_rpow2(planes[0], rfactor=2)
    planes[0] = core.resize.Spline36(planes[0], src.width, src.height)

    if only_luma and not (mask_detail or get_mask):
        return planes[0]

    if mask_detail or get_mask:
        upscaled = fvf.Resize(descaled, src.width, src.height, kernel=kernel, taps=taps, a1=b, a2=c)
        mask = eoe.rs.get_descale_mask(get_y(src), upscaled, rescale_threshold, **mask_args)
        if mask_detail:
            planes[0] = core.std.MaskedMerge(planes[0], get_y(src), mask)
    scaled = join(planes)
    scaled = core.resize.Point(scaled, format=src.format)
    return (scaled, make_similar_mask(mask, scaled)) if get_mask else scaled


# endregion setup

src = core.lsmas.LWLibavSource(src_path)
src = eoe.fmt.set(src, "16")

# denoise inside rescale
rs, rs_mask = rescale(
    src,
    1444,
    812,
    "bicubic",
    b=0,
    c=0.5,
    mask_detail=False,
    get_mask=True,
    rescale_threshold=0.05,
    morpho=False,
    maximum=4,
    inflate=2,
)

# seperate denoise for areas not rescaled
sloc = [0, 0, 0.3, 0, 0.45, 2, 0.55, 4, 0.6, 6, 0.75, 16, 1, 16]
dft = core.dfttest.DFTTest(src, tbsize=1, slocation=sloc)
# the ED is cursed
dn = eoe.dn.BM3D(dft, [2, 0], 2, "lc")

aa = lvf.aa.nnedi3(dn, opencl=True)
aa = core.rgvs.Repair(aa, dn, 17)


def select_rs(n, f: vs.VideoFrame):
    if f.props["PlaneStatsAverage"] > 0.001:
        return aa
    else:
        return rs


# select whether to rescale based on rescale mask
credits_fix = core.std.FrameEval(dn, select_rs, rs_mask.std.PlaneStats(), [dn, rs])

# ideally this would go before the frame eval, but i'm running out of vram here
knlm = core.knlm.KNLMeansCL(credits_fix, d=2, a=3, h=0.7, channels="UV")

dh = haf.FineDehalo(knlm, darkstr=0, brightstr=0.9)

deband = dumb3kdb(dh, 16, [30, 24, 24], [8, 2])

grain = kgf.adaptive_grain(deband, 0.2, luma_scaling=8)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    # debug_output(rs, "rs")
    debug_output(dn, "dn")
    debug_output(aa, "aa")
    debug_output(credits_fix, "credits_fix")
    debug_output(rs_mask.std.PlaneStats(), "rs_mask", frame_props=True)
    debug_output(knlm, "knlm")
    debug_output(dh, "dh")
    debug_output(grain, "grain")
