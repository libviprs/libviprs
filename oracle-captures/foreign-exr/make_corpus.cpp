// Corpus generator for the OpenEXR lane (issue #504).
//
// Every fixture the EXR tests and the fuzz corpus use is written HERE, by
// the OpenEXR *reference implementation* (the Homebrew `openexr` formula,
// headers and all), because that is the implementation the file format
// specification is defined against. Nothing in this file is hand-rolled
// bit-twiddling: the compression codecs, the channel layout and the
// header attributes all come from the reference writer.
//
// The expected PIXEL values are a separate question and come from
// `vips openexrload`, captured by `capture.py` in this directory. Two
// oracles, two jobs: the reference implementation says what a valid file
// looks like, and vips says what libviprs has to agree with.
//
// Build and run (from this directory):
//
//     c++ -std=c++17 -O1 make_corpus.cpp $(pkg-config --cflags --libs OpenEXR) -o /tmp/make_corpus_504
//     /tmp/make_corpus_504 fixtures
//
// Recorded tool versions live in oracle.json under meta.

#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfOutputFile.h>
#include <ImfTiledOutputFile.h>
#include <ImfDeepScanLineOutputFile.h>
#include <ImfDeepFrameBuffer.h>
#include <ImfPartType.h>
#include <ImfCompression.h>
#include <ImfPixelType.h>
#include <ImfArray.h>
#include <ImathBox.h>
#include <half.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace Imf;
using namespace Imath;

namespace {

std::string outDir;

std::string path(const std::string &name) { return outDir + "/" + name; }

// A deterministic ramp. Band `b` of pixel (x, y) is
// `(x + y * width + b * 7) * step`, so no two samples in a small image
// collide and the value is exactly representable in half for the step
// values used below.
float ramp(int x, int y, int width, int b, float step)
{
    return static_cast<float>(x + y * width + b * 7) * step;
}

// Scanline file, arbitrary channel names, one pixel type for all channels.
void writeScanline(const std::string &name,
                   const std::vector<std::string> &channels,
                   PixelType type,
                   Compression compression,
                   int width, int height,
                   int originX = 0, int originY = 0,
                   float step = 0.5f,
                   const Box2i *displayWindowOverride = nullptr)
{
    Box2i dataWindow(V2i(originX, originY),
                     V2i(originX + width - 1, originY + height - 1));
    Box2i displayWindow = displayWindowOverride ? *displayWindowOverride : dataWindow;

    Header header(displayWindow, dataWindow, 1.0f,
                  V2f(0, 0), 1.0f, INCREASING_Y, compression);

    const size_t n = channels.size();
    for (const auto &c : channels)
        header.channels().insert(c, Channel(type));

    // One contiguous interleaved plane, `n` samples per pixel.
    std::vector<half>          hbuf(static_cast<size_t>(width) * height * n);
    std::vector<float>         fbuf(static_cast<size_t>(width) * height * n);
    std::vector<unsigned int>  ubuf(static_cast<size_t>(width) * height * n);

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (size_t b = 0; b < n; ++b) {
                const size_t i = (static_cast<size_t>(y) * width + x) * n + b;
                const float v = ramp(x, y, width, static_cast<int>(b), step);
                hbuf[i] = half(v);
                fbuf[i] = v;
                ubuf[i] = static_cast<unsigned int>(x + y * width + b * 7);
            }

    FrameBuffer fb;
    const size_t sampleSize = (type == HALF) ? sizeof(half)
                            : (type == FLOAT) ? sizeof(float)
                                              : sizeof(unsigned int);
    char *base = (type == HALF)  ? reinterpret_cast<char *>(hbuf.data())
               : (type == FLOAT) ? reinterpret_cast<char *>(fbuf.data())
                                 : reinterpret_cast<char *>(ubuf.data());

    // The frame buffer base pointer is relative to pixel (0, 0) in FILE
    // coordinates, so a data window that does not start at the origin has
    // to be backed off by its own origin. This is exactly what
    // openexr2vips.c does on the read side (`imf_buffer - left - ...`).
    const size_t xStride = sampleSize * n;
    const size_t yStride = sampleSize * n * width;
    char *origin = base
                 - static_cast<ptrdiff_t>(originX) * static_cast<ptrdiff_t>(xStride)
                 - static_cast<ptrdiff_t>(originY) * static_cast<ptrdiff_t>(yStride);

    for (size_t b = 0; b < n; ++b)
        fb.insert(channels[b],
                  Slice(type, origin + b * sampleSize, xStride, yStride));

    OutputFile file(path(name).c_str(), header);
    file.setFrameBuffer(fb);
    file.writePixels(height);
    std::printf("wrote %s\n", name.c_str());
}

// Tiled file, RGB(A) half, so the tiled arm of openexr2vips.c is covered.
void writeTiled(const std::string &name,
                const std::vector<std::string> &channels,
                int width, int height, int tileW, int tileH,
                float step = 0.5f)
{
    Box2i window(V2i(0, 0), V2i(width - 1, height - 1));
    Header header(window, window, 1.0f, V2f(0, 0), 1.0f, INCREASING_Y, ZIP_COMPRESSION);

    const size_t n = channels.size();
    for (const auto &c : channels)
        header.channels().insert(c, Channel(HALF));
    header.setTileDescription(TileDescription(tileW, tileH, ONE_LEVEL));

    std::vector<half> hbuf(static_cast<size_t>(width) * height * n);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (size_t b = 0; b < n; ++b)
                hbuf[(static_cast<size_t>(y) * width + x) * n + b] =
                    half(ramp(x, y, width, static_cast<int>(b), step));

    FrameBuffer fb;
    for (size_t b = 0; b < n; ++b)
        fb.insert(channels[b],
                  Slice(HALF,
                        reinterpret_cast<char *>(hbuf.data()) + b * sizeof(half),
                        sizeof(half) * n,
                        sizeof(half) * n * width));

    TiledOutputFile file(path(name).c_str(), header);
    file.setFrameBuffer(fb);
    file.writeTiles(0, file.numXTiles() - 1, 0, file.numYTiles() - 1);
    std::printf("wrote %s (tiled %dx%d)\n", name.c_str(), tileW, tileH);
}

// A deep scanline file: one sample LIST per pixel rather than one sample.
// Neither vips nor libviprs reads deep EXR, and both have to say so rather
// than mis-parse it, so the corpus needs a real one.
void writeDeep(const std::string &name, int width, int height)
{
    Box2i window(V2i(0, 0), V2i(width - 1, height - 1));
    Header header(window, window, 1.0f, V2f(0, 0), 1.0f, INCREASING_Y, ZIPS_COMPRESSION);
    header.channels().insert("R", Channel(HALF));
    header.channels().insert("Z", Channel(FLOAT));
    header.setType(DEEPSCANLINE);

    std::vector<unsigned int> counts(static_cast<size_t>(width) * height, 1);
    std::vector<half>  rStore(static_cast<size_t>(width) * height);
    std::vector<float> zStore(static_cast<size_t>(width) * height);
    std::vector<half *>  rPtrs(static_cast<size_t>(width) * height);
    std::vector<float *> zPtrs(static_cast<size_t>(width) * height);
    for (size_t i = 0; i < rStore.size(); ++i) {
        rStore[i] = half(static_cast<float>(i) * 0.5f);
        zStore[i] = static_cast<float>(i);
        rPtrs[i] = &rStore[i];
        zPtrs[i] = &zStore[i];
    }

    DeepFrameBuffer fb;
    fb.insertSampleCountSlice(Slice(UINT,
        reinterpret_cast<char *>(counts.data()),
        sizeof(unsigned int), sizeof(unsigned int) * width));
    fb.insert("R", DeepSlice(HALF,
        reinterpret_cast<char *>(rPtrs.data()),
        sizeof(half *), sizeof(half *) * width, sizeof(half)));
    fb.insert("Z", DeepSlice(FLOAT,
        reinterpret_cast<char *>(zPtrs.data()),
        sizeof(float *), sizeof(float *) * width, sizeof(float)));

    DeepScanLineOutputFile file(path(name).c_str(), header);
    file.setFrameBuffer(fb);
    file.writePixels(height);
    std::printf("wrote %s (deep)\n", name.c_str());
}

const std::vector<std::string> RGB  = {"R", "G", "B"};
const std::vector<std::string> RGBA = {"R", "G", "B", "A"};

} // namespace

int main(int argc, char **argv)
{
    outDir = (argc > 1) ? argv[1] : "fixtures";

    // The compression sweep, all on the same 8x4 RGBA half payload so the
    // decoded pixels are identical across every lossless method and the
    // lossy ones stand out on their own.
    struct { const char *suffix; Compression c; } sweep[] = {
        {"none",   NO_COMPRESSION},
        {"rle",    RLE_COMPRESSION},
        {"zips",   ZIPS_COMPRESSION},
        {"zip",    ZIP_COMPRESSION},
        {"piz",    PIZ_COMPRESSION},
        {"pxr24",  PXR24_COMPRESSION},
        {"b44",    B44_COMPRESSION},
        {"b44a",   B44A_COMPRESSION},
        {"dwaa",   DWAA_COMPRESSION},
        {"dwab",   DWAB_COMPRESSION},
    };
    for (const auto &s : sweep)
        writeScanline(std::string("rgba_half_") + s.suffix + ".exr",
                      RGBA, HALF, s.c, 8, 4);

    // Sample types. HALF and FLOAT both widen to f32; UINT is the ceiling
    // that needs the uint carrier (#517).
    writeScanline("rgba_float_zip.exr", RGBA, FLOAT, ZIP_COMPRESSION, 8, 4);
    writeScanline("rgba_uint_zip.exr",  RGBA, UINT,  ZIP_COMPRESSION, 8, 4);

    // Channel counts.
    writeScanline("rgb_half_zip.exr",  RGB,   HALF, ZIP_COMPRESSION, 8, 4);
    writeScanline("y_half_zip.exr",    {"Y"}, HALF, ZIP_COMPRESSION, 8, 4);
    writeScanline("z_float_zip.exr",   {"Z"}, FLOAT, ZIP_COMPRESSION, 8, 4);

    // A FLOAT payload whose values are NOT representable in half, so the
    // half-carrier question is measurable rather than theoretical.
    writeScanline("rgba_float_fine.exr", RGBA, FLOAT, ZIP_COMPRESSION, 8, 4,
                  0, 0, 1.0f / 3.0f);

    // Data window off the origin, and a display window that differs from
    // the data window. vips sizes the image from the DATA window.
    writeScanline("rgba_half_offset.exr", RGBA, HALF, ZIP_COMPRESSION, 8, 4, 5, 7);
    {
        Box2i display(V2i(0, 0), V2i(15, 15));
        writeScanline("rgba_half_display.exr", RGBA, HALF, ZIP_COMPRESSION,
                      8, 4, 2, 3, 0.5f, &display);
    }

    // Tiled.
    writeTiled("rgba_half_tiled.exr", RGBA, 8, 4, 4, 2);
    writeTiled("rgba_half_tiled_ragged.exr", RGBA, 7, 5, 4, 4);

    // One pixel, the smallest legal image.
    writeScanline("rgba_half_1x1.exr", RGBA, HALF, ZIP_COMPRESSION, 1, 1);

    // Deep, which neither vips nor libviprs reads.
    writeDeep("deep_scanline.exr", 8, 4);

    return 0;
}
