// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <stdio.h>

// #include "mesh_tensor.h"

// __global__ void dilated_face_adjacencies_kernel(@ARGS_DEF) {
//     @PRECALC
//     @alias(FAF, in0)
//     int dilation = in1_shape0;
//     int N = in0_shape0;
//     int F = in0_shape1;

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int bs = idx / (F * 3);
//     int f = idx / 3 % F;
//     int k = idx % 3;

//     if (bs >= N)
//         return;

//     int a = f;
//     int b = @FAF(bs, f, k);
//     for (int d = 1; d < dilation; ++d) {
//         int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
//         a = b;
//         if ((d & 1) == 0) {     // go to next
//             b = @FAF(bs, b, i < 2 ? i + 1 : 0);
//         } else {                // go to previous
//             b = @FAF(bs, b, i > 0 ? i - 1 : 2);
//         }
//     }
//     @out(bs, f, k) = b;
// }

// dilated_face_adjacencies_kernel<<<(in0_shape0*in0_shape1*3-1)/512+1, 512>>>(@ARGS);