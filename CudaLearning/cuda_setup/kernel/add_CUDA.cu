__global__ void add_kernel(float *c,
                           const float *a,
                           const float *b,
                           int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * n + i;
    if (i < n && j < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_add(float *c,
                float *a,
                float *b,
                int n)
{
    dim3 block(16, 16);
    dim3 grid(n / 16, n / 16);
    add_kernel<<<grid, block>>>(c, a, b, n);
}