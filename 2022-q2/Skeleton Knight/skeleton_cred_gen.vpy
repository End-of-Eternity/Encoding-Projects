from pathlib import Path
import vapoursynth as vs
import EoEfunc as eoe

core = vs.core

core.num_threads = 24
core.max_cache_size = 1024 * 10

src = core.lsmas.LWLibavSource(
    r"[Erai-raws] Gaikotsu Kishi-sama Tadaima Isekai e Odekake-chuu - 01 [1080p][Multiple"
    r" Subtitle][1D9B823B].mkv"
)
ed = eoe.fmt.set(src[31889:], "rgb24")


credits_ranges: list[tuple[int, int]] = [
    (9, 78),
    (78, 156),
    (160, 230),
    (230, 337),
    (551, 628),
    (628, 709),
    (717, 826),
    (826, 908),
    (919, 984),
    (984, 1048),
    (1048, 1112),
    (1126, 1200),
    (1200, 1294),
    (1385, 1456),
    (1456, 1543),
    (1551, 1637),
    (1637, 1732),
    (1825, 1913),
    (2009, 2077),
    (2077, 2158),
]

cards: list[vs.VideoNode] = []

Path("./creds").mkdir(exist_ok=True)

for credit_range in credits_ranges:
    card = ed[credit_range[0] : credit_range[1]]
    dir = Path(f"./creds/{credit_range[0]}-{credit_range[1]}")
    dir.mkdir(exist_ok=True)
    write: vs.VideoNode = core.imwri.Write(card, "png", dir / "%03d.png")

    def _cb(node, n, result):
        print("Wrote Frame")

    for i in range(write.num_frames):
        write.get_frame_async_raw(i, _cb)
    cards.append(card)


ed_highlight_cards = ed[0]
for i, card in enumerate(cards):
    ed_highlight_cards = ed_highlight_cards + card.text.Text(i, 5)
ed_highlight_cards = ed_highlight_cards[1:]

while True:
    pass
