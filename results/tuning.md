Baseline:
Seconds: 8.97772
MiB/s: 3207.94
Speedup (best CPU): 25.9414
Speedup (best GPU): -1 (no previous GPU result)

Using raw pointers instead of shared:
Seconds: 6.10271
MiB/s: 4719.22
Speedup (best CPU): 38.1625
Speedup (best GPU): 1.47111

cuda::malloc on device only once (instead of each image), reusing memory:
Seconds: 4.93172
MiB/s: 5839.74
Speedup (best CPU): 47.2238
Speedup (best GPU): 1.23744

using uint8_t instead of pixel_t for transfer
and looking up the color in the host caused a slowdown:
Seconds: 5.8341
MiB/s: 4936.49
Speedup (best CPU): 39.9196
Speedup (best GPU): 0.845327

after adding pragma omp for pixel lookup, utilizing cpu efficiently:
Seconds: 4.51053
MiB/s: 6385.06
Speedup (best CPU): 51.6336
Speedup (best GPU): 1.09338

manual loop unrolling 
```c++
while (iteration++ < g_colors && z.norm() < g_infinity) {
    z = pfc::square(z) + c;
}
```
into 
```c++
// Unrolling the loop manually
for (uint8_t i = 0; i < g_colors; i += 4) {
    if (z.norm() >= g_infinity) break;
    z = pfc::square(z) + c;
    iteration++;

    ...
}
```
or pragma omp
```c++
#pragma omp unroll
```
did not result in a performance gain (stayed around the same). removed first iteration and initialized z with c.

using cuFloatComplex directly did not improve performance

fmaf in real, image square() improved performance a little bit (1-5%)
Seconds: 4.42474
MiB/s: 6508.85
Speedup (best CPU): 52.6347
Speedup (best GPU): 1.01939

inline calculations without complex_t performed worse (-5%)

optimizing variables (uint32, 16, etc instead of size_t) hardly improved performance

allocating host memory with cudaMallocHost improved performance by 10%
Seconds: 3.99189
MiB/s: 7214.62
Speedup (best CPU): 58.3419
Speedup (best GPU): 1.10843