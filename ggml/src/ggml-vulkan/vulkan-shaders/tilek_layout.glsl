#ifndef TILEK_LAYOUT_GLSL
#define TILEK_LAYOUT_GLSL

// TileK-first layout: an nrows x K tensor stored as K/TILEK strips,
// each holding TILEK contiguous K values for every row.
// Requires K % TILEK == 0 and TILEK == device coopmat K (checked on host).

#ifndef TILEK
#define TILEK 16
#endif

#if (TILEK == 0) || ((TILEK & (TILEK - 1)) != 0)
#error "TILEK must be a power of two"
#endif

const uint TILEK_U = uint(TILEK);

uint tilek_first_idx(const uint row, const uint kidx, const uint nrows) {
    return TILEK_U * ((kidx / TILEK_U) * nrows + row) + (kidx % TILEK_U);
}

// coopMatLoad element offset for the tile at (row0, k0); use stride = TILEK.
uint tilek_first_tile_start(const uint row0, const uint k0, const uint nrows) {
    return k0 * nrows + row0 * TILEK_U;
}

#endif // TILEK_LAYOUT_GLSL
