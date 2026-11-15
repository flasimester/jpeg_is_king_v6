"""
Core conversion engine: RAW / HEIC / TIFF / PNG / video -> JPEG, WebP, PNG or TIFF.

Can be used as a library (see app.py) or as a CLI:

    python converter.py photo.ARW folder_with_photos -q 85 -s 50 -o out_dir -r
    python converter.py shots/ --format webp -q 80
    python converter.py shots/ --format jpeg --progressive --subsampling 4:4:4
    python converter.py shots/ --format png --png-level 9
    python converter.py shots/ --format tiff --tiff-compression deflate
    python converter.py --selftest [sample.ARW sample.HEIC ...]     # check this installation
"""
from __future__ import annotations

import argparse
import errno
import io
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

# Input files are the user's own: lift Pillow's decompression-bomb limit so huge panoramas
# and scans (above ~89 million pixels) open instead of warning or failing.
Image.MAX_IMAGE_PIXELS = None

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC/HEIF unavailable
    pillow_heif = None

try:
    import rawpy
except ImportError:  # RAW unavailable
    rawpy = None

# --------------------------------------------------------------------------
# Supported formats
# --------------------------------------------------------------------------
RAW_EXTS = {".arw", ".dng", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2"}
HEIF_EXTS = {".heic", ".heif"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
ALL_EXTS = RAW_EXTS | HEIF_EXTS | IMAGE_EXTS | VIDEO_EXTS

# Output formats: key -> (PIL format name, file extension)
OUTPUT_FORMATS = {
    "jpeg": ("JPEG", ".jpg"),
    "webp": ("WEBP", ".webp"),
    "png": ("PNG", ".png"),
    "tiff": ("TIFF", ".tif"),
}
SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
AUTO = "auto"  # pick 4:4:4 or 4:2:0 per image by measuring colour damage
SUBSAMPLING_CHOICES = [AUTO] + list(SUBSAMPLING)
TIFF_COMPRESSIONS = {
    "lzw": "tiff_lzw",
    "deflate": "tiff_adobe_deflate",
    "jpeg": "jpeg",
    "none": "raw",
}

# Auto-detection tuning (see detect_subsampling)
AUTO_PIXEL_ERROR = 24  # a pixel counts as "damaged" if its colour error is above this (0-255 scale)

DEFAULT_OUT_SUBDIR = "converted"
_ALL_OUT_SUBDIRS = {"converted", "jpeg_output"}  # skipped when scanning folders

VIDEO_TIMEOUT = 120  # seconds FFmpeg may spend on one frame before we give up


# --------------------------------------------------------------------------
# Readable error messages
# --------------------------------------------------------------------------
def friendly_error(e: BaseException) -> str:
    """Turn an exception into a short message a person can act on."""
    name = type(e).__name__
    msg = str(e).strip()
    if isinstance(e, RuntimeError) and not name.startswith("LibRaw"):
        return msg or "Unknown error"  # our own messages are already readable
    if isinstance(e, FileExistsError):
        return "The output location already exists as a file (choose a different output folder)"
    if isinstance(e, FileNotFoundError):
        what = getattr(e, "filename", None)
        return f"File or folder not found: {Path(str(what)).name}" if what else "File or folder not found"
    if isinstance(e, PermissionError):
        return "Permission denied (read-only folder, or the file is open in another program)"
    if isinstance(e, OSError) and e.errno == errno.ENOSPC:
        return "Disk is full"
    if isinstance(e, OSError) and (e.errno == errno.ENAMETOOLONG or getattr(e, "winerror", None) == 206):
        return "File path is too long"
    if isinstance(e, UnidentifiedImageError):
        return "Not a readable image (corrupt, empty, or the wrong file type)"
    if isinstance(e, MemoryError):
        return "Not enough memory for this image (try a smaller Resize %)"
    if name.startswith("LibRaw"):
        return f"RAW file could not be decoded by LibRaw (unsupported camera or corrupt file): {msg[:120]}"
    if isinstance(e, ValueError) and msg:
        return msg
    return f"{name}: {msg[:160]}" if msg else name


def to_long_path(path: str) -> str:
    """Windows: prefix very long absolute paths with \\\\?\\ so they open (limit is otherwise 260 chars)."""
    if len(path) < 240 or path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):  # UNC share
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path


def _fs(path: Path):
    """Path to hand to the file system (long-path safe on Windows, unchanged elsewhere)."""
    if os.name == "nt":
        return to_long_path(os.path.abspath(str(path)))
    return path


# --------------------------------------------------------------------------
# FFmpeg lookup (works from source and from a PyInstaller bundle)
# --------------------------------------------------------------------------
def find_ffmpeg() -> str | None:
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(getattr(sys, "_MEIPASS", "")) / exe)
        candidates.append(Path(sys.executable).parent / exe)
    here = Path(__file__).resolve().parent
    candidates += [here / exe, here / "ffmpeg" / exe]
    for c in candidates:
        if c.is_file():
            return str(c)
    return shutil.which("ffmpeg")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class Loaded:
    image: Image.Image
    note: str = ""  # shown to the user, e.g. "embedded preview used"
    icc: bytes | None = None  # colour profile of the source
    mode: str = "RGB"  # mode of the source (decides whether its profile can be kept)
    pre_scale: float = 1.0  # 0.5 when a RAW was already decoded at half size


def _load_raw(path: Path, half: bool = False) -> Loaded:
    if rawpy is None:
        raise RuntimeError("RAW support is missing: the 'rawpy' package is not installed")
    with rawpy.imread(str(path)) as raw:
        try:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=half)
            return Loaded(Image.fromarray(rgb), pre_scale=0.5 if half else 1.0)
        except Exception as decode_error:  # noqa: BLE001
            # Demosaicing failed: fall back to the preview JPEG embedded in the RAW file
            try:
                thumb = raw.extract_thumb()
            except Exception:  # noqa: BLE001
                raise decode_error from None
            if thumb.format == rawpy.ThumbFormat.JPEG:
                img = Image.open(io.BytesIO(thumb.data))
                img.load()
                img = ImageOps.exif_transpose(img)
            else:
                img = Image.fromarray(thumb.data)
            return Loaded(img, note="RAW could not be decoded: embedded low-resolution preview used")


def _run_ffmpeg_frame(ffmpeg: str, path: Path, seconds: float) -> subprocess.CompletedProcess:
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if seconds > 0:
        cmd += ["-ss", f"{seconds:.3f}"]
    cmd += ["-i", str(path), "-an", "-sn", "-frames:v", "1", "-pix_fmt", "rgb24",
            "-f", "image2pipe", "-vcodec", "png", "-"]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        # stdin=DEVNULL matters: a windowed (no-console) .exe has no valid stdin handle and
        # subprocess.run would otherwise fail with "WinError 6: The handle is invalid".
        return subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                              creationflags=flags, timeout=VIDEO_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg took longer than {VIDEO_TIMEOUT} s (corrupt file or slow drive?)") from None
    except OSError as e:
        raise RuntimeError(f"FFmpeg could not be started: {e}") from None


def _last_line(data: bytes) -> str:
    lines = [ln.strip() for ln in data.decode("utf-8", "replace").splitlines() if ln.strip()]
    return lines[-1][:160] if lines else ""


def _load_video_frame(path: Path, seconds: float) -> Loaded:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found (install it, or put ffmpeg.exe next to the app)")
    note = ""
    proc = _run_ffmpeg_frame(ffmpeg, path, seconds)
    if (proc.returncode != 0 or not proc.stdout) and seconds > 0:
        # Asked for a moment after the end of the clip: use the first frame instead of failing
        retry = _run_ffmpeg_frame(ffmpeg, path, 0.0)
        if retry.returncode == 0 and retry.stdout:
            proc, note = retry, f"clip is shorter than {seconds:g} s: first frame used"
    if proc.returncode != 0 or not proc.stdout:
        reason = _last_line(proc.stderr) or "no video stream found"
        raise RuntimeError(f"FFmpeg could not read a frame: {reason}")
    img = Image.open(io.BytesIO(proc.stdout))
    img.load()
    return Loaded(img, note=note)


def load_image(path: Path, video_seconds: float = 0.0, half: bool = False) -> Loaded:
    ext = path.suffix.lower()
    if ext in RAW_EXTS:
        return _load_raw(path, half)
    if ext in VIDEO_EXTS:
        return _load_video_frame(path, video_seconds)
    if ext in HEIF_EXTS and pillow_heif is None:
        raise RuntimeError("HEIC/HEIF support is missing: the 'pillow-heif' package is not installed")
    with Image.open(path) as im:
        im.load()
        icc = im.info.get("icc_profile")
        mode = im.mode
        img = ImageOps.exif_transpose(im)  # a copy, safe to use after the file is closed
    return Loaded(img, icc=icc, mode=mode)


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------
def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def _prepare(img: Image.Image, keep_alpha: bool) -> Image.Image:
    """Normalise the mode. JPEG flattens transparency onto white; others may keep it."""
    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        img = img.point(lambda i: i * (1 / 256)).convert("L")
    if _has_alpha(img):
        rgba = img.convert("RGBA")
        if keep_alpha:
            return rgba
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def chroma_damage(img: Image.Image) -> float:
    """Fraction of pixels whose colour would change visibly under 4:2:0 subsampling.

    Simulates what JPEG does (halve the colour resolution, then scale it back up)
    and counts the pixels whose Cb/Cr colour error exceeds AUTO_PIXEL_ERROR.
    Smooth photos score close to 0; thin coloured lines, coloured text and
    screenshots score high.
    """
    from PIL import ImageChops

    w, h = img.size
    if w < 2 or h < 2:
        return 0.0
    _, cb, cr = img.convert("YCbCr").split()
    half = ((w + 1) // 2, (h + 1) // 2)
    diffs = []
    for ch in (cb, cr):
        approx = ch.resize(half, Image.Resampling.BOX).resize((w, h), Image.Resampling.BILINEAR)
        diffs.append(ImageChops.difference(ch, approx))
    worst = ImageChops.lighter(diffs[0], diffs[1])
    hist = worst.histogram()
    return sum(hist[AUTO_PIXEL_ERROR + 1:]) / float(w * h)


def damage_limit(sensitivity: int) -> float:
    """Damage fraction above which an image gets 4:4:4. Sensitivity 1-100:
    higher = more eager to choose 4:4:4 (1 -> ~10 %, 50 -> 1 %, 100 -> 0.1 %)."""
    s = max(1, min(100, int(sensitivity)))
    return 10 ** (-1 - 2 * s / 100)


def detect_subsampling(img: Image.Image, sensitivity: int = 50) -> tuple[str, float]:
    """Return ("4:4:4" | "4:2:0", damage_fraction) for an RGB image."""
    dmg = chroma_damage(img)
    return ("4:4:4" if dmg > damage_limit(sensitivity) else "4:2:0"), dmg


@dataclass
class Result:
    src: Path
    status: str  # "ok" | "skipped" | "error"
    dst: Path | None = None
    message: str = ""
    in_size: int = 0
    out_size: int = 0


def plan_output(src: Path, out_dir: Path | None, used: set[str], fmt: str = "jpeg") -> Path:
    """Work out the destination path, avoiding clashes inside one batch."""
    out_ext = OUTPUT_FORMATS[fmt][1]
    folder = Path(out_dir) if out_dir else src.parent / DEFAULT_OUT_SUBDIR
    suffix = "_frame" if src.suffix.lower() in VIDEO_EXTS else ""
    dst = folder / f"{src.stem}{suffix}{out_ext}"
    try:
        if dst.resolve() == src.resolve():
            dst = folder / f"{src.stem}_converted{out_ext}"
    except OSError:
        pass
    base, n = dst, 1
    while str(dst).lower() in used:
        dst = base.with_name(f"{base.stem}_{n}{out_ext}")
        n += 1
    used.add(str(dst).lower())
    return dst


def build_save_kwargs(
    fmt: str,
    quality: int,
    progressive: bool,
    subsampling: str,
    optimize: bool,
    lossless: bool,
    png_level: int,
    tiff_compression: str,
) -> dict:
    """Translate the user-facing options into Pillow save() arguments."""
    pil_fmt = OUTPUT_FORMATS[fmt][0]
    kw: dict = {"format": pil_fmt}
    if fmt == "jpeg":
        kw.update(quality=int(quality), optimize=bool(optimize), progressive=bool(progressive))
        if subsampling in SUBSAMPLING:
            kw["subsampling"] = SUBSAMPLING[subsampling]
    elif fmt == "webp":
        kw.update(quality=int(quality), lossless=bool(lossless), method=6)
    elif fmt == "png":
        # Note: Pillow's optimize=True forces level 9 and would override png_level,
        # so PNG is controlled by compress_level alone.
        kw.update(compress_level=max(0, min(9, int(png_level))))
    elif fmt == "tiff":
        kw["compression"] = TIFF_COMPRESSIONS.get(tiff_compression, "tiff_lzw")
        if kw["compression"] == "jpeg":
            kw["quality"] = int(quality)
    return kw


def _atomic_save(img: Image.Image, dst: Path, kwargs: dict) -> None:
    """Write to a temporary file and rename it into place, so an interrupted or failed save
    can never leave a half-written file that later runs would treat as already converted."""
    tmp = dst.with_name(dst.name + ".part")
    try:
        img.save(_fs(tmp), **kwargs)  # format= is always given, so the .part suffix is harmless
        os.replace(_fs(tmp), _fs(dst))
    except BaseException:
        try:
            os.remove(_fs(tmp))
        except OSError:
            pass
        raise


def convert_file(
    src: Path,
    out_dir: Path | None = None,
    quality: int = 85,
    resize_pct: int = 100,
    overwrite: bool = False,
    keep_exif: bool = True,
    video_seconds: float = 0.0,
    used: set[str] | None = None,
    fmt: str = "jpeg",
    progressive: bool = False,
    subsampling: str = AUTO,
    optimize: bool = True,
    lossless: bool = False,
    png_level: int = 6,
    tiff_compression: str = "lzw",
    sensitivity: int = 50,
) -> Result:
    src = Path(src)
    used = used if used is not None else set()
    try:
        if fmt not in OUTPUT_FORMATS:
            raise ValueError(f"Unknown output format '{fmt}'")
        if src.is_dir():
            raise ValueError("This is a folder, not a file")
        if not src.is_file():
            raise FileNotFoundError(2, "not found", str(src))
        in_size = src.stat().st_size
        if in_size == 0:
            raise ValueError("The file is empty (0 bytes)")
        dst = plan_output(src, out_dir, used, fmt)
        if dst.exists() and dst.stat().st_size > 0 and not overwrite:
            return Result(src, "skipped", dst, "already exists", in_size, dst.stat().st_size)

        ext = src.suffix.lower()
        half = ext in RAW_EXTS and resize_pct <= 50 and rawpy is not None  # RAW: decode at half size (4x faster)
        loaded = load_image(src, video_seconds, half)
        img = loaded.image

        exif_bytes = b""
        if keep_exif and fmt in ("jpeg", "webp", "png") and ext not in RAW_EXTS | VIDEO_EXTS:
            try:
                exif = img.getexif()
                if exif:
                    exif[0x0112] = 1  # orientation already applied
                    exif_bytes = exif.tobytes()
            except Exception:  # noqa: BLE001
                exif_bytes = b""

        img = _prepare(img, keep_alpha=(fmt != "jpeg"))
        scale = resize_pct / 100.0 / loaded.pre_scale
        if abs(scale - 1.0) > 1e-9:
            w, h = img.size
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)

        chroma_note = ""
        if fmt == "jpeg":
            if subsampling == AUTO:
                subsampling, _dmg = detect_subsampling(img, sensitivity)
                chroma_note = f" {subsampling} auto"
            else:
                chroma_note = f" {subsampling}"

        try:
            os.makedirs(_fs(dst.parent), exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Cannot create the output folder: {friendly_error(e)}") from None
        kwargs = build_save_kwargs(fmt, quality, progressive, subsampling, optimize,
                                   lossless, png_level, tiff_compression)
        if exif_bytes:
            kwargs["exif"] = exif_bytes
        # Keep the source's colour profile (e.g. iPhone Display P3) so colours don't shift.
        # Skipped for CMYK / greyscale / 16-bit sources, whose profile doesn't describe RGB output.
        if loaded.icc and loaded.mode in ("RGB", "RGBA", "P"):
            kwargs["icc_profile"] = loaded.icc
        _atomic_save(img, dst, kwargs)

        message = f"{img.size[0]}x{img.size[1]}{chroma_note}"
        if loaded.note:
            message += f" · {loaded.note}"
        return Result(src, "ok", dst, message, in_size, dst.stat().st_size)
    except Exception as e:  # noqa: BLE001
        return Result(src, "error", None, friendly_error(e))


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------
def collect_files(paths, recursive: bool = True) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()

    def add(p: Path):
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            found.append(p)

    for raw_p in paths:
        p = Path(raw_p)
        if p.is_file():
            if p.suffix.lower() in ALL_EXTS:
                add(p)
        elif p.is_dir():
            if recursive:
                for root, dirs, files in os.walk(p):
                    dirs[:] = sorted(d for d in dirs if d not in _ALL_OUT_SUBDIRS)
                    for f in sorted(files):
                        fp = Path(root) / f
                        if fp.suffix.lower() in ALL_EXTS:
                            add(fp)
            else:
                for fp in sorted(p.iterdir()):
                    if fp.is_file() and fp.suffix.lower() in ALL_EXTS:
                        add(fp)
    return found


def human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


# --------------------------------------------------------------------------
# Self-test: what actually works on THIS computer / THIS build
# --------------------------------------------------------------------------
def _sample_picture(w: int = 320, h: int = 240) -> Image.Image:
    from PIL import ImageDraw

    img = Image.new("RGB", (w, h), (30, 60, 120))
    d = ImageDraw.Draw(img)
    d.ellipse((60, 40, 260, 200), fill=(220, 80, 40))
    d.line((0, 0, w, h), fill=(255, 255, 255), width=3)
    return img


def _ffmpeg_version(ffmpeg: str) -> str:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    p = subprocess.run([ffmpeg, "-version"], stdin=subprocess.DEVNULL, capture_output=True,
                       creationflags=flags, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {p.returncode}")
    return p.stdout.decode("utf-8", "replace").splitlines()[0][:100]


def selftest(extra_files=()) -> list[tuple[str, str, str]]:
    """Run real conversions on generated samples. Returns (check, "PASS"|"FAIL"|"SKIP", detail) rows.
    Any extra_files (e.g. a real .ARW or .HEIC from your camera or phone) are converted too."""
    rows: list[tuple[str, str, str]] = []

    def check(name, fn):
        try:
            status, detail = fn()
        except Exception as e:  # noqa: BLE001
            status, detail = "FAIL", friendly_error(e)
        rows.append((name, status, detail))

    import PIL

    frozen = getattr(sys, "frozen", False)
    rows.append(("Environment", "PASS",
                 f"Python {sys.version.split()[0]} · Pillow {PIL.__version__} · "
                 f"{'packaged .exe' if frozen else 'running from source'} · {sys.platform}"))

    with tempfile.TemporaryDirectory(prefix="imgconv_selftest_") as tmp:
        tmpdir = Path(tmp)
        out = tmpdir / "out"
        sample = _sample_picture()
        png = tmpdir / "sample.png"
        sample.save(png)

        # --- output formats
        def out_fmt(fmt):
            def run():
                r = convert_file(png, out, overwrite=True, fmt=fmt)
                if r.status != "ok" or r.dst is None:
                    return "FAIL", r.message
                with Image.open(r.dst) as im:
                    if im.size != sample.size:
                        return "FAIL", f"wrong size {im.size}"
                return "PASS", f"{human(r.in_size)} -> {human(r.out_size)}"
            return run

        for fmt in OUTPUT_FORMATS:
            check(f"Output: {fmt.upper()}", out_fmt(fmt))

        # --- input image formats
        def in_fmt(ext, pil_name):
            def run():
                p = tmpdir / f"in_{ext.lstrip('.')}{ext}"
                sample.save(p, format=pil_name)
                r = convert_file(p, out, overwrite=True)
                return ("PASS", r.message) if r.status == "ok" else ("FAIL", r.message)
            return run

        for ext, pil_name in ((".jpg", "JPEG"), (".png", "PNG"), (".tif", "TIFF"), (".webp", "WEBP"),
                              (".bmp", "BMP")):
            check(f"Input: {ext}", in_fmt(ext, pil_name))

        # --- colour profile kept
        def icc_check():
            from PIL import ImageCms

            prof = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            p = tmpdir / "profile.png"
            sample.save(p, icc_profile=prof)
            r = convert_file(p, out, overwrite=True)
            if r.status != "ok" or r.dst is None:
                return "FAIL", r.message
            with Image.open(r.dst) as im:
                return ("PASS", "colour profile kept") if im.info.get("icc_profile") else ("FAIL", "profile lost")

        check("Colour profile (ICC)", icc_check)

        # --- HEIC / HEIF
        def heic_check():
            if pillow_heif is None:
                return "SKIP", "pillow-heif is not installed: HEIC/HEIF files cannot be opened"
            p = tmpdir / "sample.heic"
            try:
                sample.save(p, format="HEIF")
            except Exception as e:  # noqa: BLE001
                return "SKIP", f"library loaded (v{pillow_heif.__version__}) but cannot create a test file: {e}"
            r = convert_file(p, out, overwrite=True)
            return ("PASS", f"pillow-heif {pillow_heif.__version__}: {r.message}") if r.status == "ok" \
                else ("FAIL", r.message)

        check("HEIC / HEIF", heic_check)

        # --- RAW
        def raw_check():
            if rawpy is None:
                return "SKIP", "rawpy is not installed: RAW files (ARW, DNG ...) cannot be opened"
            return "PASS", (f"rawpy {getattr(rawpy, '__version__', '?')}, LibRaw "
                            f"{getattr(rawpy, 'libraw_version', '?')} loaded "
                            f"(pass a real sample file to test decoding)")

        check("RAW library", raw_check)

        # --- FFmpeg + video containers
        ffmpeg = find_ffmpeg()

        def ffmpeg_check():
            if not ffmpeg:
                return "SKIP", "FFmpeg not found: video files cannot be read"
            return "PASS", f"{ffmpeg} · {_ffmpeg_version(ffmpeg)}"

        check("FFmpeg", ffmpeg_check)

        def make_clip(ext):
            p = tmpdir / f"clip{ext}"
            cmd = [ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2",
                   "-c:v", "mpeg4", "-pix_fmt", "yuv420p"]
            if ext == ".m4v":
                cmd += ["-f", "mp4"]
            cmd.append(str(p))
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            pr = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, creationflags=flags, timeout=60)
            if pr.returncode != 0 or not p.exists():
                raise RuntimeError("could not create a test clip: " + _last_line(pr.stderr))
            return p

        def video_check(ext):
            def run():
                if not ffmpeg:
                    return "SKIP", "FFmpeg not found"
                clip = make_clip(ext)
                r = convert_file(clip, out, overwrite=True)
                if r.status != "ok":
                    return "FAIL", r.message
                r2 = convert_file(clip, out, overwrite=True, video_seconds=1.0)
                if r2.status != "ok":
                    return "FAIL", f"frame at 1 s: {r2.message}"
                r3 = convert_file(clip, out, overwrite=True, video_seconds=30.0)  # beyond the end
                if r3.status != "ok":
                    return "FAIL", f"frame past the end: {r3.message}"
                return "PASS", "first frame, frame at 1 s, and past-the-end fallback all OK"
            return run

        for ext in (".mp4", ".mov", ".m4v", ".avi", ".mkv"):
            check(f"Video: {ext}", video_check(ext))

        # --- error handling: bad input must give a message, never a crash
        def bad_input():
            bad = []
            for name, data in (("empty.png", b""), ("garbage.png", b"definitely not an image"),
                               ("empty.mp4", b""), ("garbage.heic", b"nope"), ("garbage.arw", b"nope")):
                p = tmpdir / name
                p.write_bytes(data)
                r = convert_file(p, out, overwrite=True)
                if r.status != "error" or not r.message:
                    bad.append(f"{name} -> {r.status}")
            r = convert_file(tmpdir / "missing.jpg", out)
            if r.status != "error":
                bad.append("missing file not reported")
            if bad:
                return "FAIL", "; ".join(bad)
            return "PASS", "corrupt, empty and missing files each give a readable error message"

        check("Error handling", bad_input)

        # --- your own sample files
        for f in extra_files:
            f = Path(f)

            def run(f=f):
                r = convert_file(f, out, overwrite=True)
                return ("PASS", f"{r.message} ({human(r.in_size)} -> {human(r.out_size)})") \
                    if r.status == "ok" else ("FAIL", r.message)

            check(f"Sample: {f.name}", run)
    return rows


def format_report(rows) -> str:
    width = max(len(r[0]) for r in rows)
    lines = ["Image Converter self-test", "=" * 25]
    for name, status, detail in rows:
        lines.append(f"[{status}] {name.ljust(width)}  {detail}")
    n = {s: sum(1 for r in rows if r[1] == s) for s in ("PASS", "FAIL", "SKIP")}
    lines += ["", f"Result: {n['PASS']} passed, {n['FAIL']} failed, {n['SKIP']} skipped"]
    if n["SKIP"]:
        lines.append("Skipped items are formats this computer / build cannot handle (see the detail).")
    return "\n".join(lines)


def run_selftest_cli(extra_files=()) -> int:
    """Print the report (when there is a console) and also save selftest.txt next to the program."""
    rows = selftest(extra_files)
    text = format_report(rows)
    try:
        if sys.stdout:  # a windowed .exe has no console
            print(text)
    except Exception:  # noqa: BLE001
        pass
    folder = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
    try:
        (folder / "selftest.txt").write_text(text + "\n", encoding="utf-8")
    except OSError:
        pass
    return 1 if any(r[1] == "FAIL" for r in rows) else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert RAW/HEIC/TIFF/PNG/video to JPEG, WebP, PNG or TIFF")
    ap.add_argument("inputs", nargs="*", help="files and/or folders")
    ap.add_argument("-o", "--out", help=f"output folder (default: a '{DEFAULT_OUT_SUBDIR}' folder next to each file)")
    ap.add_argument("-f", "--format", choices=list(OUTPUT_FORMATS), default="jpeg", help="output format (default jpeg)")
    ap.add_argument("-q", "--quality", type=int, default=85, help="quality 1-100 for JPEG/WebP/TIFF-jpeg (default 85)")
    ap.add_argument("-s", "--resize", type=int, default=100, help="resize percent 10-100 (default 100)")
    ap.add_argument("-r", "--recursive", action="store_true", help="scan folders recursively")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing output files")
    ap.add_argument("--no-exif", action="store_true", help="do not copy EXIF data")
    ap.add_argument("--video-time", type=float, default=0.0, help="video frame position in seconds (default 0)")
    ap.add_argument("--progressive", action="store_true", help="JPEG: progressive encoding")
    ap.add_argument("--subsampling", choices=SUBSAMPLING_CHOICES, default=AUTO,
                    help="JPEG: chroma subsampling. 'auto' (default) picks 4:4:4 or 4:2:0 per image; "
                         "4:4:4 = best colour, biggest file")
    ap.add_argument("--sensitivity", type=int, default=50,
                    help="JPEG auto subsampling: 1-100, higher = more images get 4:4:4 (default 50)")
    ap.add_argument("--no-optimize", action="store_true", help="JPEG: disable the extra optimisation pass")
    ap.add_argument("--lossless", action="store_true", help="WebP: lossless mode (quality then means effort)")
    ap.add_argument("--png-level", type=int, default=6, help="PNG: zlib compression level 0-9 (default 6)")
    ap.add_argument("--tiff-compression", choices=list(TIFF_COMPRESSIONS), default="lzw",
                    help="TIFF: compression method (default lzw)")
    ap.add_argument("--selftest", action="store_true",
                    help="check this installation (formats, FFmpeg, RAW / HEIC libraries); "
                         "any inputs given are converted as extra samples")
    a = ap.parse_args(argv)

    if a.selftest:
        return run_selftest_cli(a.inputs)
    if not a.inputs:
        ap.error("give at least one file or folder (or use --selftest)")

    quality = max(1, min(100, a.quality))
    resize = max(10, min(100, a.resize))
    files = collect_files(a.inputs, a.recursive)
    if not files:
        print("No supported files found.")
        return 1

    used: set[str] = set()
    errors = 0
    for i, f in enumerate(files, 1):
        r = convert_file(
            f, Path(a.out) if a.out else None, quality, resize,
            a.overwrite, not a.no_exif, a.video_time, used,
            fmt=a.format, progressive=a.progressive, subsampling=a.subsampling,
            optimize=not a.no_optimize, lossless=a.lossless,
            png_level=a.png_level, tiff_compression=a.tiff_compression,
            sensitivity=max(1, min(100, a.sensitivity)),
        )
        tag = {"ok": "OK  ", "skipped": "SKIP", "error": "FAIL"}[r.status]
        extra = f"{human(r.in_size)} -> {human(r.out_size)}" if r.status == "ok" else r.message
        print(f"[{i}/{len(files)}] {tag} {f.name}  {extra}")
        errors += r.status == "error"
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
