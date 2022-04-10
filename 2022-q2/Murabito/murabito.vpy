import vapoursynth as vs
import EoEfunc as eoe
import lvsfunc as lvf
from vardefunc.deband import dumb3kdb
import kagefunc as kgf
import havsfunc as haf

# region setup

DEFAULT_PATH = (
    r"Shijou Saikyou no Daimaou, Murabito A ni Tensei suru - 01 - 1080p WEB x264 -NanDesuKa"
    r" (CR).mkv"
)
DEFAULT_CACHE = 32
DEFAULT_THREADS = 24

env = eoe.setup_env(globals())
core: vs._CoreProxy = env["core"]
src_path: str = env["src_path"]
GPU: int = env["GPU"]
debug: bool = env["debug"]


def rescale_f(
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

    sloc = [0, 1, 0.3, 1, 0.45, 2, 0.55, 4, 0.6, 6, 0.75, 16, 1, 16]
    planes[0] = core.dfttest.DFTTest(planes[0], tbsize=1, slocation=sloc)
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

rescale, rescale_mask = rescale_f(
    src,
    1440,
    810,
    "bicubic",
    b=0,
    c=1 / 2,
    mask_detail=True,
    rescale_threshold=0.03,
    morpho=False,
    maximum=3,
    inflate=3,
    get_mask=True,
)

knlm = core.knlm.KNLMeansCL(rescale, d=2, a=3, h=0.6, channels="UV")

aa = lvf.aa.nnedi3(knlm, opencl=True)
aa = core.std.MaskedMerge(aa, knlm, rescale_mask)

dehalo = haf.FineDehalo(aa, darkstr=0, brightstr=0.8, thlima=112)

deband = dumb3kdb(dehalo, 16, [28, 16, 16], [48, 4])

grain = kgf.adaptive_grain(deband, 0.25, luma_scaling=4)

final = grain
final = eoe.fmt.set(final, "10")
final.set_output()

if debug:
    debug_output = eoe.debug_output
    debug_output(src, "src")
    debug_output(
        core.lsmas.LWLibavSource(
            "[Judas] Shijou Saikyou no Daimaou, Murabito A ni Tensei suru - S01E01 INCOMPLETE.mkv"
        )
    )
    debug_output(
        core.lsmas.LWLibavSource(
            "[EMBER] Shijou Saikyou no Daimaou, Murabito A ni Tensei suru - 01.mkv"
        ),
        "ember",
    )
    debug_output(rescale, "rescale")
    debug_output(rescale_mask, "rescale_mask")
    # debug_output(dft, "dft")
    # debug_output(bm3d, "bm3d")
    debug_output(knlm, "knlm")
    debug_output(aa, "aa")
    debug_output(deband, "deband")
    debug_output(grain, "grain")
