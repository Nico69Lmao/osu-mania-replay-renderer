from pathlib import Path
import struct


class OsuDbReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, size):
        if self.pos + size > len(self.data):
            raise EOFError

        value = self.data[self.pos:self.pos + size]
        self.pos += size
        return value

    def u8(self):
        return self.read(1)[0]

    def i16(self):
        return struct.unpack_from("<h", self.read(2))[0]

    def i32(self):
        return struct.unpack_from("<i", self.read(4))[0]

    def i64(self):
        return struct.unpack_from("<q", self.read(8))[0]

    def f32(self):
        return struct.unpack_from("<f", self.read(4))[0]

    def f64(self):
        return struct.unpack_from("<d", self.read(8))[0]

    def boolean(self):
        return self.u8() != 0

    def uleb128(self):
        result = 0
        shift = 0

        while True:
            byte = self.u8()
            result |= (byte & 0x7f) << shift

            if byte & 0x80 == 0:
                return result

            shift += 7

    def string(self):
        marker = self.u8()

        if marker == 0:
            return ""

        if marker != 0x0b:
            raise ValueError(f"unexpected string marker {marker}")

        length = self.uleb128()
        return self.read(length).decode("utf-8", errors="ignore")

    def skip_timing_points(self):
        count = self.i32()

        for _ in range(count):
            self.f64()
            self.f64()
            self.boolean()

    def star_ratings(self):
        count = self.i32()
        values = {}

        for _ in range(count):
            self.u8()
            mods = self.i32()
            marker = self.u8()

            if marker == 0x0c:
                stars = self.f32()
            elif marker == 0x0d:
                stars = self.f64()
            else:
                raise ValueError(f"unexpected star rating marker {marker}")

            values[mods] = stars

        return values

    def beatmap_entry(self, version):
        entry_size = None

        if version < 20191106:
            entry_size = self.i32()
            start = self.pos

        artist = self.string()
        artist_unicode = self.string()
        title = self.string()
        title_unicode = self.string()
        creator = self.string()
        difficulty = self.string()
        audio_file = self.string()
        md5_hash = self.string()
        osu_file = self.string()

        self.u8()
        self.i16()
        self.i16()
        self.i16()
        self.i64()

        if version < 20140609:
            self.u8()
            self.u8()
            self.u8()
            self.u8()
        else:
            self.f32()
            self.f32()
            self.f32()
            self.f32()

        self.f64()

        ratings = [
            self.star_ratings(),
            self.star_ratings(),
            self.star_ratings(),
            self.star_ratings(),
        ]

        self.i32()
        self.i32()
        self.i32()
        self.skip_timing_points()

        beatmap_id = self.i32()
        beatmap_set_id = self.i32()

        if entry_size is not None:
            self.pos = start + entry_size
        else:
            self.i32()
            self.u8()
            self.u8()
            self.u8()
            self.u8()
            self.i16()
            self.f32()
            self.u8()
            self.i64()
            self.boolean()
            self.string()
            self.i64()
            self.boolean()
            self.string()
            self.i64()
            self.boolean()
            self.i64()
            self.i32()
            self.i64()

        return {
            "artist": artist_unicode or artist,
            "title": title_unicode or title,
            "difficulty": difficulty,
            "md5": md5_hash,
            "beatmap_id": beatmap_id,
            "beatmap_set_id": beatmap_set_id,
            "ratings": ratings,
        }


def find_osu_db(start_path):
    path = Path(start_path).resolve()

    for parent in [path.parent, *path.parents]:
        candidate = parent / "osu!.db"

        if candidate.exists():
            return candidate

    return None


def read_mania_star_rating(osu_file, md5_hash, mods_int):
    db_path = find_osu_db(osu_file)

    if db_path is None:
        return None

    data = db_path.read_bytes()

    rating = read_mania_star_rating_by_hash_scan(data, md5_hash, mods_int)

    if rating is not None:
        return rating

    reader = OsuDbReader(data)

    try:
        version = reader.i32()
        reader.i32()
        reader.boolean()
        reader.i64()
        reader.string()
        beatmap_count = reader.i32()

        for _ in range(beatmap_count):
            entry = reader.beatmap_entry(version)

            if entry["md5"].lower() != md5_hash.lower():
                continue

            mania_ratings = entry["ratings"][3]

            if mods_int in mania_ratings:
                return mania_ratings[mods_int]

            if mods_int & 512 and (mods_int & ~512 | 64) in mania_ratings:
                return mania_ratings[mods_int & ~512 | 64]

            return mania_ratings.get(0)
    except Exception:
        return None

    return None


def select_star_rating(ratings, mods_int):
    if mods_int in ratings:
        return ratings[mods_int]

    if mods_int & 512:
        nc_as_dt = (mods_int & ~512) | 64

        if nc_as_dt in ratings:
            return ratings[nc_as_dt]

    return ratings.get(0)


def read_mania_star_rating_by_hash_scan(data, md5_hash, mods_int):
    needle = md5_hash.encode("ascii")
    start = 0

    while True:
        index = data.find(needle, start)

        if index < 0:
            return None

        reader = OsuDbReader(data)
        reader.pos = index + len(needle)

        try:
            reader.string()
            reader.u8()
            reader.i16()
            reader.i16()
            reader.i16()
            reader.i64()
            reader.f32()
            reader.f32()
            reader.f32()
            reader.f32()
            reader.f64()
            reader.star_ratings()
            reader.star_ratings()
            reader.star_ratings()
            mania = reader.star_ratings()
            rating = select_star_rating(mania, mods_int)

            if rating is not None:
                return rating
        except Exception:
            pass

        start = index + 1
