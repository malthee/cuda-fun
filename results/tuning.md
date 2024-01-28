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