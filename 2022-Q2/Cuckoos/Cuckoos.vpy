import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
import havsfunc as haf
from debandshit import placebo_deband

# region setup

DEFAULT_PATH = r"[SubsPlease] Kakkou no Iinazuke - 01 (1080p) [46444F07].mkv"
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

    # here are the only changes to rescale
    planes[0] = eoe.dn.BM3D(planes[0], [1.3, 0], 0, "fast")
    planes[0] = lvf.aa.nnedi3(planes[0], True)

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

# I need to add a "filter" function to rescale so that i dont have to copy paste the code in every
# time i want to do something whilst descaled.
rs, rs_mask = rescale(
    src,
    1552,
    873,
    "bicubic",
    b=0,
    c=0.5,
    mask_detail=True,
    morpho=False,
    rescale_threshold=0.03,
    get_mask=True,
)

knlm = core.knlm.KNLMeansCL(rs, d=1, a=3, h=0.6, channels="UV")

dehalo = haf.FineDehalo(knlm, 2, darkstr=0, brightstr=0.6)

deband = placebo_deband(dehalo, 16, 2, 1, 2)

final = deband
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output(rs, "rs")
    debug_output(rs_mask, "rs_mask")
    debug_output(knlm, "knlm")
    debug_output(dehalo, "dehalo")
    debug_output(deband, "deband")
