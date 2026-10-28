// The products the local stacks' projections share, on cooperative matrices: C^T = W X^T for an
// f16 weight `w` of K columns and the M tokens `x`, `[M][K]` in f16, both eight to a uvec4, in
// tiles of 16 x 16 multiplied in f16 and summed in f32. w is in tiles of 16 rows by 16 columns,
// each row-major, a row of tiles' one after another (`local::tiled`). A subgroup takes a unit of
// RT tiles of 16 rows of w, which it loads from memory as the A operand DEPTH steps of 16 columns
// ahead of the products, against every tile of 16 of its workgroup's block of BM tokens, which
// the workgroup stages in shared memory KC columns at a time as the B operand, zero past M. Each
// tile of C^T is 16 rows of w by 16 tokens.
//
// WorkGroupID.z is the block of tokens, and WorkGroupID.y the part of the PARTS that cut the
// columns of w, each summed apart: what the sums' order depends on, which no other choice here
// changes, so that a token's products do not depend on what else is in its block.
//
// The including kernel declares K, KC, M, BM, RT, DEPTH, UNITS and PARTS and the `x` and `w`
// buffers, then includes this, defines `unit_row` and `epilogue`, and calls `run`.
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require

#define TILE_A coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define TILE_B coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define TILE_C coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>

// The token tiles of a block, and the tiles of C^T a subgroup sums at once.
const uint TT = BM / 16;
const uint TILES = RT * TT;
// The columns of a part, and its steps of 16 and chunks of KC.
const uint PART = K / PARTS;
const uint STEPS = PART / 16;
const uint CHUNKS = PART / KC;
const uint CHUNK_STEPS = KC / 16;
// A staged token's f16, padded by one uvec4, so that the lanes reading a tile's 16 tokens at once
// read from different banks.
const uint STRIDE = KC + 8;
// The uvec4 of a chunk of the block each invocation stages.
const uint STAGED = (BM * (KC / 8) + WORKGROUP - 1) / WORKGROUP;

shared uvec4 xs[BM * (STRIDE / 8)];

// The first row of w of tile r of unit `unit`: the including kernel's.
uint unit_row(uint unit, uint r);

// What the including kernel makes of the tiles of C^T of unit `unit` in token block `block`, tile
// r of the rows by tile t of the tokens at acc[r * TT + t], over part WorkGroupID.y of the columns.
void epilogue(uint unit, uint block, TILE_C acc[TILES]);

// Writes chunk `chunk` of part `part` of the block's tokens into xs, `[BM][STRIDE]`.
void stage(uint block, uint part, uint chunk) {
    uvec4 tokens[STAGED];
    [[unroll]] for (uint i = 0; i < STAGED; i++) {
        uint v = i * WORKGROUP + gl_LocalInvocationID.x;
        uint m = block * BM + v / (KC / 8);
        tokens[i] = uvec4(0);
        if (v < BM * (KC / 8) && m < M) {
            tokens[i] = x[m * (K / 8) + (part * PART + chunk * KC) / 8 + v % (KC / 8)];
        }
    }
    [[unroll]] for (uint i = 0; i < STAGED; i++) {
        uint v = i * WORKGROUP + gl_LocalInvocationID.x;
        if (v < BM * (KC / 8)) {
            xs[v / (KC / 8) * (STRIDE / 8) + v % (KC / 8)] = tokens[i];
        }
    }
}

void load_tiles(uint rows[RT], uint part, uint s, out TILE_A a[RT]) {
    [[unroll]] for (uint r = 0; r < RT; r++) {
        uint at = (rows[r] / 16 * (K / 16) + part * PART / 16 + s) * 32;
        coopMatLoad(a[r], w, at, 2, gl_CooperativeMatrixLayoutRowMajor);
    }
}

// Adds the products of step `s` of the part, `a` the unit's tiles of w there, with the staged
// tokens to acc.
void multiply(uint s, TILE_A a[RT], inout TILE_C acc[TILES]) {
    uint column = s % CHUNK_STEPS * 16;
    TILE_B b[TT];
    [[unroll]] for (uint t = 0; t < TT; t++) {
        coopMatLoad(b[t], xs, (t * 16 * STRIDE + column) / 8, STRIDE / 8,
            gl_CooperativeMatrixLayoutColumnMajor);
    }
    [[unroll]] for (uint r = 0; r < RT; r++) {
        [[unroll]] for (uint t = 0; t < TT; t++) {
            acc[r * TT + t] = coopMatMulAdd(a[r], b[t], acc[r * TT + t]);
        }
    }
}

// The unit's products over the part, its tiles of w loaded DEPTH steps ahead of their use, into
// the registers of the step whose products have just been taken: loaded anywhere else, the
// compiler copies them and waits for every load to copy them. Past the last step the last is
// loaded again, unused, so that no branch parts the steps. When the part is staged chunk by
// chunk, every subgroup of the workgroup restages each chunk together.
void sum_unit(uint unit, uint block, uint part) {
    uint rows[RT];
    [[unroll]] for (uint r = 0; r < RT; r++) {
        rows[r] = unit_row(unit, r);
    }
    TILE_C acc[TILES];
    [[unroll]] for (uint i = 0; i < TILES; i++) {
        acc[i] = TILE_C(0.0);
    }
    TILE_A ahead[DEPTH][RT];
    [[unroll]] for (uint d = 0; d < DEPTH; d++) {
        load_tiles(rows, part, d, ahead[d]);
    }
    for (uint chunk = 0; chunk < CHUNKS; chunk++) {
        if (CHUNKS > 1) {
            barrier();
            stage(block, part, chunk);
            barrier();
        }
        for (uint first = chunk * CHUNK_STEPS; first < (chunk + 1) * CHUNK_STEPS;
             first += DEPTH) {
            [[unroll]] for (uint d = 0; d < DEPTH; d++) {
                multiply(first + d, ahead[d], acc);
                load_tiles(rows, part, min(first + d + DEPTH, STEPS - 1), ahead[d]);
            }
        }
    }
    epilogue(unit, block, acc);
}

// Every unit the workgroup takes. A part staged at once is staged once, and each of the
// workgroup's subgroups then takes every NumWorkGroups.x-th of the subgroups' worth of units from
// its own on, apart from the others. One staged chunk by chunk is restaged for every
// NumWorkGroups.x-th of the workgroups' worth of units, all the subgroups together, of which
// UNITS is then a multiple.
void run() {
    uint part = gl_WorkGroupID.y;
    uint block = gl_WorkGroupID.z;
    if (CHUNKS == 1) {
        stage(block, part, 0);
        barrier();
        uint stride = gl_NumWorkGroups.x * SUBGROUPS;
        for (uint unit = gl_WorkGroupID.x * SUBGROUPS + gl_SubgroupID; unit < UNITS;
             unit += stride) {
            sum_unit(unit, block, part);
        }
    } else {
        for (uint group = gl_WorkGroupID.x; group * SUBGROUPS < UNITS;
             group += gl_NumWorkGroups.x) {
            sum_unit(group * SUBGROUPS + gl_SubgroupID, block, part);
        }
    }
}
