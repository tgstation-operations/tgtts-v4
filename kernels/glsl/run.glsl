// The run of positions the kernels over one take, read on every run so that a recording serves
// every run of its rows: positions start to start + len - 1 of slot `slot` of the caches, rows 0
// to len - 1 of each tensor. A tensor may have more rows, which the kernels leave as they are.
layout(std430, binding = 0) readonly buffer Run {
    uint start;
    uint len;
    uint slot;
};
