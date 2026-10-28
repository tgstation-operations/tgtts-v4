// Where each of BATCH rows runs, read on every run so that a recording serves every placement
// of its rows: `at` holds each row's position, then each row's cache slot when there are SLOTS
// of them. A row at a position past the cache is spare: it writes nothing into a cache, and
// attends to nothing.
//
// The including kernel declares BATCH, SLOTS, CAPACITY and the `at` buffer, then includes this.

// Row b's position and slot, and whether it runs.
bool placed(uint b, out uint pos, out uint slot) {
    pos = at_[b];
    slot = SLOTS > 1 ? at_[BATCH + b] : 0;
    return pos < CAPACITY && slot < SLOTS;
}
