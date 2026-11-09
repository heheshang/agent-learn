-- ===========================================================================
-- 15-agent-sql 种子数据库
--
-- 表结构参考 LangChain SQL Agent 教程使用的 Chinook（数字媒体商店），
-- 但数据换成中文，方便用自然语言提问。
--
-- 本文件由 data/make_seed.py 生成 —— 请改生成脚本，不要手改这里。
-- ===========================================================================

PRAGMA foreign_keys = ON;

-- ── 表结构 ────────────────────────────────────────────────

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

CREATE INDEX IF NOT EXISTS idx_track_genre ON Track(GenreId);
CREATE INDEX IF NOT EXISTS idx_album_artist ON Album(ArtistId);
CREATE INDEX IF NOT EXISTS idx_invoice_customer ON Invoice(CustomerId);

-- ── 数据 ──────────────────────────────────────────────────
BEGIN TRANSACTION;

-- 流派
INSERT INTO Genre (GenreId, Name) VALUES (1, '摇滚');
INSERT INTO Genre (GenreId, Name) VALUES (2, '流行');
INSERT INTO Genre (GenreId, Name) VALUES (3, '民谣');
INSERT INTO Genre (GenreId, Name) VALUES (4, '爵士');
INSERT INTO Genre (GenreId, Name) VALUES (5, '古典');
INSERT INTO Genre (GenreId, Name) VALUES (6, '电子');
INSERT INTO Genre (GenreId, Name) VALUES (7, '金属');
INSERT INTO Genre (GenreId, Name) VALUES (8, '蓝调');

-- 艺人
INSERT INTO Artist (ArtistId, Name, Country) VALUES (1, '陈绮贞', '台湾');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (2, '李志', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (3, '窦唯', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (4, '崔健', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (5, '朴树', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (6, '许巍', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (7, '张三', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (8, '五月天', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (9, '周杰伦', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (10, '林俊杰', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (11, '邓丽君', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (12, '王菲', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (13, 'Beyond', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (14, '窦靖童', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (15, '王菲', '中国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (16, 'Nirvana', '美国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (17, 'Miles Davis', '美国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (18, 'Bach', '德国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (19, 'Daft Punk', '法国');
INSERT INTO Artist (ArtistId, Name, Country) VALUES (20, 'Radiohead', '英国');

INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (1, '时间的歌', 1, 2013);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (1, '航线', 1, 4, 318172, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (2, '路灯', 1, 1, 363849, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (3, '慢车', 1, 7, 321274, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (4, '花火', 1, 2, 180077, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (5, '空白', 1, 5, 355193, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (6, '逆光', 1, 8, 221578, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (2, '还是会寂寞', 1, 2014);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (7, '侧影', 2, 1, 331475, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (8, '散步', 2, 6, 312964, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (9, '未完成的诗', 2, 7, 428214, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (10, '潮汐', 2, 3, 201710, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (11, '轨迹', 2, 8, 225301, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (12, '重逢', 2, 8, 330142, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (3, '关于郑州的记忆', 2, 2020);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (13, '光年', 3, 2, 191276, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (14, '第七日', 3, 6, 273345, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (15, '循环', 3, 8, 239477, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (16, '深呼吸', 3, 5, 317160, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (17, '沉默', 3, 2, 212868, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (18, '候鸟', 3, 4, 251059, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (4, '梵高先生', 2, 2019);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (19, '雨后', 4, 2, 200114, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (20, '归途', 4, 8, 286862, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (21, '钟摆', 4, 8, 296723, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (22, '旧信封', 4, 8, 300089, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (23, '影子', 4, 4, 383101, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (5, '天空之城', 2, 2010);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (24, '回声', 5, 6, 381884, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (25, '晨光', 5, 2, 256138, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (26, '午夜', 5, 2, 215319, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (27, '海岸线', 5, 6, 381630, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (28, '远方', 5, 5, 187651, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (29, '独白', 5, 4, 260821, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (6, '黑梦', 3, 2024);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (30, '夜色', 6, 4, 352445, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (31, '未命名曲目 31', 6, 7, 319316, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (32, '未命名曲目 32', 6, 1, 226166, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (33, '未命名曲目 33', 6, 6, 345593, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (7, '高级动物', 3, 2013);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (34, '未命名曲目 34', 7, 4, 301829, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (35, '未命名曲目 35', 7, 4, 252425, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (36, '未命名曲目 36', 7, 2, 261116, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (37, '未命名曲目 37', 7, 5, 189620, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (38, '未命名曲目 38', 7, 4, 330571, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (39, '未命名曲目 39', 7, 2, 267684, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (8, '山河水', 3, 2019);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (40, '未命名曲目 40', 8, 6, 390489, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (41, '未命名曲目 41', 8, 5, 186084, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (42, '未命名曲目 42', 8, 5, 303830, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (43, '未命名曲目 43', 8, 2, 187849, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (44, '未命名曲目 44', 8, 1, 311830, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (45, '未命名曲目 45', 8, 8, 338794, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (9, '一无所有', 4, 2019);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (46, '未命名曲目 46', 9, 1, 338996, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (47, '未命名曲目 47', 9, 1, 316145, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (48, '未命名曲目 48', 9, 8, 272672, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (49, '未命名曲目 49', 9, 1, 284543, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (50, '未命名曲目 50', 9, 6, 376197, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (10, '新长征路上的摇滚', 4, 2014);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (51, '未命名曲目 51', 10, 7, 470970, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (52, '未命名曲目 52', 10, 6, 353801, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (53, '未命名曲目 53', 10, 8, 333667, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (11, '生如夏花', 5, 2023);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (54, '未命名曲目 54', 11, 6, 301469, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (55, '未命名曲目 55', 11, 7, 312907, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (56, '未命名曲目 56', 11, 3, 212499, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (57, '未命名曲目 57', 11, 2, 278384, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (58, '未命名曲目 58', 11, 2, 268736, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (59, '未命名曲目 59', 11, 8, 232511, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (12, '平凡之路', 5, 2010);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (60, '未命名曲目 60', 12, 2, 229521, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (61, '未命名曲目 61', 12, 3, 253366, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (62, '未命名曲目 62', 12, 5, 281543, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (63, '未命名曲目 63', 12, 1, 328918, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (64, '未命名曲目 64', 12, 8, 317922, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (65, '未命名曲目 65', 12, 6, 364268, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (13, '那些花儿', 5, 2008);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (66, '未命名曲目 66', 13, 3, 276510, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (67, '未命名曲目 67', 13, 6, 298849, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (68, '未命名曲目 68', 13, 1, 244767, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (69, '未命名曲目 69', 13, 6, 341156, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (14, '蓝莲花', 6, 2014);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (70, '未命名曲目 70', 14, 1, 320850, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (71, '未命名曲目 71', 14, 5, 321954, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (72, '未命名曲目 72', 14, 5, 244446, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (73, '未命名曲目 73', 14, 3, 269664, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (74, '未命名曲目 74', 14, 7, 345311, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (75, '未命名曲目 75', 14, 6, 288704, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (15, '曾经的你', 6, 2006);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (76, '未命名曲目 76', 15, 7, 389498, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (77, '未命名曲目 77', 15, 5, 307671, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (78, '未命名曲目 78', 15, 8, 279025, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (16, '旅行', 6, 2003);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (79, '未命名曲目 79', 16, 1, 345734, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (80, '未命名曲目 80', 16, 3, 208248, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (81, '未命名曲目 81', 16, 5, 262520, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (82, '未命名曲目 82', 16, 7, 423814, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (83, '未命名曲目 83', 16, 2, 250717, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (17, '张三的歌', 7, 2014);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (84, '未命名曲目 84', 17, 6, 385677, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (85, '未命名曲目 85', 17, 4, 395022, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (86, '未命名曲目 86', 17, 5, 299435, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (18, '山之语', 7, 2007);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (87, '未命名曲目 87', 18, 7, 444636, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (88, '未命名曲目 88', 18, 7, 310320, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (89, '未命名曲目 89', 18, 4, 408031, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (90, '未命名曲目 90', 18, 5, 357578, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (91, '未命名曲目 91', 18, 6, 391150, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (19, '倔强', 8, 2010);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (92, '未命名曲目 92', 19, 8, 314024, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (93, '未命名曲目 93', 19, 6, 388335, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (94, '未命名曲目 94', 19, 3, 213476, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (20, '突然好想你', 8, 2003);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (95, '未命名曲目 95', 20, 5, 219553, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (96, '未命名曲目 96', 20, 5, 209314, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (97, '未命名曲目 97', 20, 3, 230766, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (98, '未命名曲目 98', 20, 1, 324739, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (21, '温柔', 8, 2006);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (99, '未命名曲目 99', 21, 6, 317683, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (100, '未命名曲目 100', 21, 2, 202507, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (101, '未命名曲目 101', 21, 7, 477951, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (22, '青花瓷', 9, 2005);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (102, '未命名曲目 102', 22, 2, 264306, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (103, '未命名曲目 103', 22, 4, 257553, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (104, '未命名曲目 104', 22, 5, 290678, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (23, '七里香', 9, 2005);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (105, '未命名曲目 105', 23, 7, 433529, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (106, '未命名曲目 106', 23, 1, 376031, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (107, '未命名曲目 107', 23, 7, 300877, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (108, '未命名曲目 108', 23, 5, 239219, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (109, '未命名曲目 109', 23, 1, 236759, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (24, '稻香', 9, 2023);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (110, '未命名曲目 110', 24, 2, 248541, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (111, '未命名曲目 111', 24, 6, 266516, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (112, '未命名曲目 112', 24, 6, 396787, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (113, '未命名曲目 113', 24, 2, 263702, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (25, '江南', 10, 2005);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (114, '未命名曲目 114', 25, 1, 361224, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (115, '未命名曲目 115', 25, 2, 214817, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (116, '未命名曲目 116', 25, 6, 378553, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (117, '未命名曲目 117', 25, 5, 319463, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (118, '未命名曲目 118', 25, 5, 346965, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (26, '曹操', 10, 2024);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (119, '未命名曲目 119', 26, 4, 357272, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (120, '未命名曲目 120', 26, 1, 359063, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (121, '未命名曲目 121', 26, 8, 327897, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (122, '未命名曲目 122', 26, 3, 212936, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (27, '小酒窝', 10, 2013);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (123, '未命名曲目 123', 27, 1, 354466, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (124, '未命名曲目 124', 27, 5, 278142, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (125, '未命名曲目 125', 27, 4, 280449, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (126, '未命名曲目 126', 27, 5, 331977, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (127, '未命名曲目 127', 27, 1, 367472, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (128, '未命名曲目 128', 27, 1, 263855, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (28, '月亮代表我的心', 11, 2019);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (129, '未命名曲目 129', 28, 2, 239243, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (130, '未命名曲目 130', 28, 6, 367102, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (131, '未命名曲目 131', 28, 2, 271469, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (132, '未命名曲目 132', 28, 2, 240648, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (133, '未命名曲目 133', 28, 6, 269729, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (134, '未命名曲目 134', 28, 6, 276565, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (29, '甜蜜蜜', 11, 2015);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (135, '未命名曲目 135', 29, 2, 204783, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (136, '未命名曲目 136', 29, 6, 390736, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (137, '未命名曲目 137', 29, 6, 274866, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (30, '我只在乎你', 11, 2017);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (138, '未命名曲目 138', 30, 6, 367530, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (139, '未命名曲目 139', 30, 3, 315242, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (140, '未命名曲目 140', 30, 2, 222939, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (141, '未命名曲目 141', 30, 3, 223551, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (142, '未命名曲目 142', 30, 2, 252775, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (143, '未命名曲目 143', 30, 5, 337412, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (31, '红豆', 12, 2016);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (144, '未命名曲目 144', 31, 2, 205955, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (145, '未命名曲目 145', 31, 6, 409265, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (146, '未命名曲目 146', 31, 2, 194970, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (147, '未命名曲目 147', 31, 1, 227623, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (148, '未命名曲目 148', 31, 3, 276284, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (32, '传奇', 12, 2016);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (149, '未命名曲目 149', 32, 5, 191168, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (150, '未命名曲目 150', 32, 1, 302968, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (151, '未命名曲目 151', 32, 7, 375368, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (152, '未命名曲目 152', 32, 5, 260040, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (153, '未命名曲目 153', 32, 8, 236655, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (33, '我愿意', 12, 2010);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (154, '未命名曲目 154', 33, 5, 246626, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (155, '未命名曲目 155', 33, 5, 325866, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (156, '未命名曲目 156', 33, 7, 429935, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (34, '海阔天空', 13, 2011);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (157, '未命名曲目 157', 34, 4, 389489, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (158, '未命名曲目 158', 34, 1, 251051, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (159, '未命名曲目 159', 34, 6, 288872, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (160, '未命名曲目 160', 34, 2, 239713, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (35, '光辉岁月', 13, 2006);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (161, '未命名曲目 161', 35, 6, 404741, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (162, '未命名曲目 162', 35, 4, 265489, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (163, '未命名曲目 163', 35, 4, 390939, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (36, '真的爱你', 13, 2018);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (164, '未命名曲目 164', 36, 8, 223785, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (165, '未命名曲目 165', 36, 4, 270351, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (166, '未命名曲目 166', 36, 6, 381125, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (167, '未命名曲目 167', 36, 7, 446283, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (168, '未命名曲目 168', 36, 4, 267659, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (37, 'Blue Flamingo', 14, 2011);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (169, '未命名曲目 169', 37, 3, 257792, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (170, '未命名曲目 170', 37, 3, 281057, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (171, '未命名曲目 171', 37, 1, 337219, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (172, '未命名曲目 172', 37, 1, 333831, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (173, '未命名曲目 173', 37, 7, 474864, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (174, '未命名曲目 174', 37, 1, 299173, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (38, 'My Days', 14, 2005);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (175, '未命名曲目 175', 38, 8, 294337, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (176, '未命名曲目 176', 38, 5, 227990, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (177, '未命名曲目 177', 38, 3, 304042, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (39, '流年', 15, 2012);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (178, '未命名曲目 178', 39, 4, 382414, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (179, '未命名曲目 179', 39, 3, 245212, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (180, '未命名曲目 180', 39, 2, 274026, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (181, '未命名曲目 181', 39, 5, 261606, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (182, '未命名曲目 182', 39, 7, 393623, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (183, '未命名曲目 183', 39, 2, 273747, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (40, 'Smells Like Teen Spirit', 16, 2013);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (184, '未命名曲目 184', 40, 2, 270926, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (185, '未命名曲目 185', 40, 3, 291772, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (186, '未命名曲目 186', 40, 8, 235206, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (187, '未命名曲目 187', 40, 6, 335400, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (188, '未命名曲目 188', 40, 3, 236038, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (41, 'Come As You Are', 16, 2004);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (189, '未命名曲目 189', 41, 7, 447028, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (190, '未命名曲目 190', 41, 6, 326109, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (191, '未命名曲目 191', 41, 4, 349951, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (192, '未命名曲目 192', 41, 3, 209063, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (193, '未命名曲目 193', 41, 7, 386016, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (42, 'So What', 17, 2004);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (194, '未命名曲目 194', 42, 6, 327561, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (195, '未命名曲目 195', 42, 2, 229480, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (196, '未命名曲目 196', 42, 2, 218661, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (197, '未命名曲目 197', 42, 8, 305897, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (198, '未命名曲目 198', 42, 3, 224151, 0.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (43, 'Take Five', 17, 2008);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (199, '未命名曲目 199', 43, 7, 345737, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (200, '未命名曲目 200', 43, 5, 279886, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (201, '未命名曲目 201', 43, 6, 309716, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (202, '未命名曲目 202', 43, 6, 337436, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (203, '未命名曲目 203', 43, 6, 267503, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (204, '未命名曲目 204', 43, 1, 345700, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (44, 'Brandenburg Concerto No.3', 18, 2003);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (205, '未命名曲目 205', 44, 1, 369926, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (206, '未命名曲目 206', 44, 1, 372970, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (207, '未命名曲目 207', 44, 8, 338180, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (208, '未命名曲目 208', 44, 6, 328141, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (209, '未命名曲目 209', 44, 2, 249540, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (210, '未命名曲目 210', 44, 5, 183287, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (45, 'Air on the G String', 18, 2015);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (211, '未命名曲目 211', 45, 8, 337074, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (212, '未命名曲目 212', 45, 3, 317473, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (213, '未命名曲目 213', 45, 6, 391033, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (214, '未命名曲目 214', 45, 6, 357408, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (215, '未命名曲目 215', 45, 5, 330430, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (46, 'Around the World', 19, 2010);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (216, '未命名曲目 216', 46, 1, 378818, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (217, '未命名曲目 217', 46, 8, 233042, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (218, '未命名曲目 218', 46, 6, 354248, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (47, 'One More Time', 19, 2017);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (219, '未命名曲目 219', 47, 4, 263586, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (220, '未命名曲目 220', 47, 6, 269566, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (221, '未命名曲目 221', 47, 2, 186644, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (222, '未命名曲目 222', 47, 1, 227707, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (223, '未命名曲目 223', 47, 8, 307593, 1.99);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (48, 'Creep', 20, 2017);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (224, '未命名曲目 224', 48, 7, 325956, 1.29);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (225, '未命名曲目 225', 48, 1, 379595, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (226, '未命名曲目 226', 48, 5, 190308, 1.29);
INSERT INTO Album (AlbumId, Title, ArtistId, ReleaseYear) VALUES (49, 'Karma Police', 20, 2011);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (227, '未命名曲目 227', 49, 6, 308594, 1.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (228, '未命名曲目 228', 49, 3, 282871, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (229, '未命名曲目 229', 49, 1, 323523, 0.99);
INSERT INTO Track (TrackId, Name, AlbumId, GenreId, Milliseconds, UnitPrice) VALUES (230, '未命名曲目 230', 49, 8, 237694, 0.99);

INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (1, '芳', '黄', '法国', '黄芳1@example.com');
INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (2, '芳', '张', '美国', '张芳2@example.com');
INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (3, '静', '杨', '德国', '杨静3@studio.cn');
INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (4, '敏', '吴', '英国', '吴敏4@mail.com');
INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (5, '强', '黄', '日本', '黄强5@mail.com');
INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (6, '静', '周', '德国', '周静6@example.com');
INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (7, '强', '吴', '日本', '吴强7@studio.cn');
INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (8, '敏', '王', '日本', '王敏8@studio.cn');
INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (9, '敏', '李', '中国', '李敏9@mail.com');
INSERT INTO Customer (CustomerId, FirstName, LastName, Country, Email) VALUES (10, '娜', '黄', '英国', '黄娜10@example.com');

INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (1, 6, '2025-07-03', 2.28);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (2, 7, '2025-01-13', 0.99);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (3, 9, '2025-04-03', 3.28);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (4, 4, '2025-09-07', 2.28);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (5, 9, '2025-04-09', 3.27);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (6, 6, '2025-12-09', 1.99);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (7, 2, '2025-07-19', 4.97);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (8, 6, '2025-01-24', 6.26);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (9, 2, '2025-04-20', 6.26);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (10, 2, '2025-09-23', 3.27);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (11, 1, '2025-02-11', 3.57);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (12, 7, '2025-03-05', 0.99);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (13, 2, '2025-11-19', 4.86);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (14, 8, '2025-09-08', 2.28);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (15, 8, '2025-04-13', 3.28);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (16, 7, '2025-02-05', 6.96);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (17, 5, '2025-05-15', 3.98);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (18, 7, '2025-12-05', 3.28);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (19, 3, '2025-12-10', 6.26);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (20, 10, '2025-02-24', 1.29);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (21, 9, '2025-12-18', 3.57);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (22, 5, '2025-04-02', 3.57);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (23, 10, '2025-09-11', 4.86);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (24, 4, '2025-08-09', 3.28);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (25, 10, '2025-10-05', 0.99);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (26, 3, '2025-03-04', 4.56);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (27, 10, '2025-10-13', 2.98);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (28, 4, '2025-02-08', 6.26);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (29, 8, '2025-12-05', 4.56);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (30, 10, '2025-10-11', 4.27);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (31, 1, '2025-04-06', 3.97);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (32, 4, '2025-12-06', 1.98);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (33, 7, '2025-03-28', 2.98);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (34, 6, '2025-01-01', 1.99);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (35, 2, '2025-05-17', 2.98);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (36, 1, '2025-06-11', 4.26);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (37, 8, '2025-08-11', 4.27);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (38, 8, '2025-04-24', 3.97);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (39, 10, '2025-03-10', 3.28);
INSERT INTO Invoice (InvoiceId, CustomerId, InvoiceDate, Total) VALUES (40, 1, '2025-01-20', 5.26);

INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (1, 1, 12, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (2, 1, 207, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (3, 2, 202, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (4, 3, 86, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (5, 3, 96, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (6, 4, 185, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (7, 4, 191, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (8, 5, 46, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (9, 5, 18, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (10, 5, 178, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (11, 6, 99, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (12, 7, 74, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (13, 7, 132, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (14, 7, 50, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (15, 8, 168, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (16, 8, 127, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (17, 8, 173, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (18, 8, 116, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (19, 9, 135, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (20, 9, 110, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (21, 9, 95, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (22, 9, 10, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (23, 10, 59, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (24, 10, 122, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (25, 10, 200, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (26, 11, 57, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (27, 11, 170, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (28, 11, 47, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (29, 12, 30, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (30, 13, 93, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (31, 13, 146, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (32, 13, 83, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (33, 13, 117, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (34, 14, 63, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (35, 14, 212, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (36, 15, 224, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (37, 15, 190, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (38, 16, 25, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (39, 16, 153, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (40, 16, 140, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (41, 16, 2, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (42, 17, 80, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (43, 17, 132, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (44, 18, 17, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (45, 18, 203, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (46, 19, 83, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (47, 19, 50, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (48, 19, 153, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (49, 19, 47, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (50, 20, 212, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (51, 21, 143, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (52, 21, 131, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (53, 21, 111, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (54, 22, 60, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (55, 22, 162, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (56, 22, 47, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (57, 23, 143, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (58, 23, 155, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (59, 23, 70, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (60, 23, 152, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (61, 24, 189, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (62, 24, 204, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (63, 25, 88, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (64, 26, 20, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (65, 26, 131, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (66, 26, 67, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (67, 26, 102, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (68, 27, 95, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (69, 27, 194, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (70, 28, 209, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (71, 28, 36, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (72, 28, 94, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (73, 28, 17, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (74, 29, 68, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (75, 29, 88, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (76, 29, 171, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (77, 29, 8, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (78, 30, 212, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (79, 30, 229, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (80, 30, 13, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (81, 31, 58, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (82, 31, 164, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (83, 31, 225, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (84, 32, 60, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (85, 32, 185, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (86, 33, 55, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (87, 33, 138, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (88, 34, 81, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (89, 35, 79, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (90, 35, 158, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (91, 36, 66, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (92, 36, 166, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (93, 36, 137, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (94, 36, 226, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (95, 37, 45, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (96, 37, 85, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (97, 37, 55, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (98, 38, 9, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (99, 38, 208, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (100, 38, 56, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (101, 39, 89, 1.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (102, 39, 197, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (103, 40, 135, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (104, 40, 148, 1.29, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (105, 40, 87, 0.99, 1);
INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity) VALUES (106, 40, 112, 1.99, 1);

COMMIT;
