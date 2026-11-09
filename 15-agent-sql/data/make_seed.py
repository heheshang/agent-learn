#!/usr/bin/env python3
"""生成 15-agent-sql 的种子数据库（data/seed.sql）。

为什么要用「生成脚本 + 文本 SQL」而不是直接提交一个 `.db` 二进制文件？

1. **可再生**：`.db` 是二进制，git diff 看不出改了什么；
   文本 SQL 一眼就能审阅「这张表有哪些列、塞了什么数据」。
2. **仓库体积**：二进制文件进 git 后每次改动都是整文件变更。
3. **可移植**：换数据库引擎（Postgres）时 SQL 几乎不用改。

所以本项目的做法是：仓库里存 `seed.sql`，启动时用 rusqlite 执行它
生成真正的 SQLite 库（默认落在 `data/chinook.db`，已被 .gitignore 忽略）。

数据口径参考 LangChain SQL Agent 教程用的 Chinook（数字媒体商店），
但表名沿用英文（真实 SQL 库惯例），数据用中文 ——
这样「用自然语言提问」才有真实的中文语义可查。
"""

import pathlib
import random

random.seed(20261108)

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "seed.sql"

# ── 流派：不同流派的曲目时长分布不同 ──────────────────────────────
# 「时长」单位毫秒。金属/电子偏长，古典/爵士偏短 —— 让
# 「哪个流派平均曲目最长」这个问题有确定的答案。
GENRES = [
    ("摇滚", (220_000, 380_000)),
    ("流行", (180_000, 280_000)),
    ("民谣", (200_000, 320_000)),
    ("爵士", (240_000, 420_000)),
    ("古典", (180_000, 360_000)),
    ("电子", (260_000, 420_000)),
    ("金属", (300_000, 480_000)),
    ("蓝调", (200_000, 340_000)),
]

# ── 艺人 → 专辑 → 曲目 ────────────────────────────────────────────
ARTISTS = [
    ("陈绮贞", "台湾", ["时间的歌", "还是会寂寞"]),
    ("李志", "中国", ["关于郑州的记忆", "梵高先生", "天空之城"]),
    ("窦唯", "中国", ["黑梦", "高级动物", "山河水"]),
    ("崔健", "中国", ["一无所有", "新长征路上的摇滚"]),
    ("朴树", "中国", ["生如夏花", "平凡之路", "那些花儿"]),
    ("许巍", "中国", ["蓝莲花", "曾经的你", "旅行"]),
    ("张三", "中国", ["张三的歌", "山之语"]),
    ("五月天", "中国", ["倔强", "突然好想你", "温柔"]),
    ("周杰伦", "中国", ["青花瓷", "七里香", "稻香"]),
    ("林俊杰", "中国", ["江南", "曹操", "小酒窝"]),
    ("邓丽君", "中国", ["月亮代表我的心", "甜蜜蜜", "我只在乎你"]),
    ("王菲", "中国", ["红豆", "传奇", "我愿意"]),
    ("Beyond", "中国", ["海阔天空", "光辉岁月", "真的爱你"]),
    ("窦靖童", "中国", ["Blue Flamingo", "My Days"]),
    ("王菲", "中国", ["流年"]),
    ("Nirvana", "美国", ["Smells Like Teen Spirit", "Come As You Are"]),
    ("Miles Davis", "美国", ["So What", "Take Five"]),
    ("Bach", "德国", ["Brandenburg Concerto No.3", "Air on the G String"]),
    ("Daft Punk", "法国", ["Around the World", "One More Time"]),
    ("Radiohead", "英国", ["Creep", "Karma Police"]),
]

TRACK_TITLES = [
    "夜色", "独白", "远方", "海岸线", "午夜", "晨光", "回声", "影子",
    "旧信封", "钟摆", "归途", "雨后", "候鸟", "沉默", "深呼吸", "循环",
    "第七日", "光年", "重逢", "轨迹", "潮汐", "未完成的诗", "散步", "侧影",
    "逆光", "空白", "花火", "慢车", "路灯", "航线",
]

COUNTRIES = ["中国", "台湾", "美国", "日本", "德国", "法国", "英国"]
FIRST_NAMES = ["伟", "芳", "娜", "敏", "静", "磊", "强", "洋", "艳", "勇"]
LAST_NAMES = ["王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴"]
EMAIL_DOMAINS = ["example.com", "mail.com", "studio.cn"]


def esc(text: str) -> str:
    return text.replace("'", "''")


def build() -> str:
    lines: list[str] = []
    add = lines.append

    add("-- ===========================================================================")
    add("-- 15-agent-sql 种子数据库")
    add("--")
    add("-- 表结构参考 LangChain SQL Agent 教程使用的 Chinook（数字媒体商店），")
    add("-- 但数据换成中文，方便用自然语言提问。")
    add("--")
    add("-- 本文件由 data/make_seed.py 生成 —— 请改生成脚本，不要手改这里。")
    add("-- ===========================================================================")
    add("")
    add("PRAGMA foreign_keys = ON;")
    add("")

    # ── 表结构 ──────────────────────────────────────────────────────
    add("-- ── 表结构 ────────────────────────────────────────────────")
    add("""
CREATE TABLE IF NOT EXISTS Genre (
    GenreId   INTEGER PRIMARY KEY,
    Name      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Artist (
    ArtistId  INTEGER PRIMARY KEY,
    Name      TEXT NOT NULL,
    Country   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Album (
    AlbumId    INTEGER PRIMARY KEY,
    Title      TEXT NOT NULL,
    ArtistId   INTEGER NOT NULL REFERENCES Artist(ArtistId),
    ReleaseYear INTEGER
);

CREATE TABLE IF NOT EXISTS Track (
    TrackId     INTEGER PRIMARY KEY,
    Name        TEXT NOT NULL,
    AlbumId     INTEGER NOT NULL REFERENCES Album(AlbumId),
    GenreId     INTEGER NOT NULL REFERENCES Genre(GenreId),
    Milliseconds INTEGER NOT NULL,
    UnitPrice   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS Customer (
    CustomerId INTEGER PRIMARY KEY,
    FirstName  TEXT NOT NULL,
    LastName   TEXT NOT NULL,
    Country    TEXT NOT NULL,
    Email      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Invoice (
    InvoiceId   INTEGER PRIMARY KEY,
    CustomerId  INTEGER NOT NULL REFERENCES Customer(CustomerId),
    InvoiceDate TEXT NOT NULL,
    Total       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS InvoiceLine (
    InvoiceLineId INTEGER PRIMARY KEY,
    InvoiceId     INTEGER NOT NULL REFERENCES Invoice(InvoiceId),
    TrackId       INTEGER NOT NULL REFERENCES Track(TrackId),
    UnitPrice     REAL NOT NULL,
    Quantity      INTEGER NOT NULL
);
""")
    add("CREATE INDEX IF NOT EXISTS idx_track_genre ON Track(GenreId);")
    add("CREATE INDEX IF NOT EXISTS idx_album_artist ON Album(ArtistId);")
    add("CREATE INDEX IF NOT EXISTS idx_invoice_customer ON Invoice(CustomerId);")
    add("")

    # ── 数据 ──────────────────────────────────────────────────────
    add("-- ── 数据 ──────────────────────────────────────────────────")
    add("BEGIN TRANSACTION;")
    add("")

    add("-- 流派")
    for gid, (name, _) in enumerate(GENRES, start=1):
        add(f"INSERT INTO Genre (GenreId, Name) VALUES ({gid}, '{esc(name)}');")
    add("")

    add("-- 艺人")
    artist_rows = []
    for aid, (name, country, _) in enumerate(ARTISTS, start=1):
        artist_rows.append((aid, name, country))
        add(f"INSERT INTO Artist (ArtistId, Name, Country) VALUES ({aid}, '{esc(name)}', '{esc(country)}');")
    add("")

    # 专辑 + 曲目
    album_rows = []
    track_rows = []
    album_id = 1
    track_id = 1
    title_pool = list(TRACK_TITLES)

    for aid, (name, _country, albums) in enumerate(ARTISTS, start=1):
        for album_title in albums:
            year = random.randint(2003, 2024)
            album_rows.append((album_id, album_title, aid, year))
            add(
                f"INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) "
                f"VALUES ({album_id}, '{esc(album_title)}', {aid}, {year});"
            )

            # 曲目数：3~6 首
            n_tracks = random.randint(3, 6)
            for _ in range(n_tracks):
                title = title_pool.pop() if title_pool else f"未命名曲目 {track_id}"
                genre_id = random.randint(1, len(GENRES))
                lo, hi = GENRES[genre_id - 1][1]
                ms = random.randint(lo, hi)
                price = round(random.choice([0.99, 1.29, 1.99]), 2)
                track_rows.append((track_id, title, album_id, genre_id, ms, price))
                add(
                    f"INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) "
                    f"VALUES ({track_id}, '{esc(title)}', {album_id}, {genre_id}, {ms}, {price});"
                )
                track_id += 1

            album_id += 1

    add("")

    # 客户
    customer_rows = []
    for cid in range(1, 11):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        country = random.choice(COUNTRIES)
        email = f"{last}{first}{cid}@{random.choice(EMAIL_DOMAINS)}"
        customer_rows.append((cid, first, last, country, email))
        add(
            f"INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) "
            f"VALUES ({cid}, '{first}', '{last}', '{country}', '{email}');"
        )
    add("")

    # 发票 + 销售明细
    invoice_rows = []
    invoice_line_rows = []
    invoice_id = 1
    line_id = 1
    for i in range(1, 41):
        cid = random.randint(1, 10)
        date = f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        items = []
        total = 0.0
        n_lines = random.randint(1, 4)
        for _ in range(n_lines):
            tid = random.randint(1, len(track_rows))
            price = track_rows[tid - 1][5]
            qty = 1
            items.append((line_id, invoice_id, tid, price, qty))
            total += price * qty
            line_id += 1
        total = round(total, 2)
        invoice_rows.append((invoice_id, cid, date, total))
        invoice_line_rows.extend(items)
        invoice_id += 1

    for inv_id, cid, date, total in invoice_rows:
        add(
            f"INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) "
            f"VALUES ({inv_id}, {cid}, '{date}', {total});"
        )
    add("")
    for lid, inv_id, tid, price, qty in invoice_line_rows:
        add(
            f"INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) "
            f"VALUES ({lid}, {inv_id}, {tid}, {price}, {qty});"
        )

    add("")
    add("COMMIT;")
    add("")
    return "\n".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"已生成 {OUT}")
    print(f"  流派 {len(GENRES)} 个")
    print(f"  艺人 {len(ARTISTS)} 位")
    print(f"  专辑 {len(ARTISTS) and sum(len(a[2]) for a in ARTISTS)} 张")
    print(f"  客户 10 位 / 发票 40 条")


if __name__ == "__main__":
    main()
