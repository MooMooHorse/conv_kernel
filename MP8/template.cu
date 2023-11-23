#include <wb.h>

#define BLOCK_SIZE 256 //@@ You can change this

#define wbCheck(stmt)                                                      \
    do                                                                     \
    {                                                                      \
        cudaError_t err = stmt;                                            \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
            return -1;                                                     \
        }                                                                  \
    } while (0)

__global__ void spmvJDSKernel(float *out, int *matColStart, int *matCols,
                              int *matRowPerm, int *matRows,
                              float *matData, float *vec, int dim, int ndata) {
    //@@ insert spmv kernel for jds format

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < dim) {
        float dot = 0;
        unsigned int sec = 0;
        int sec_end = sec == matRows[0] - 1 ? dim : matColStart[sec + 1];
        int sec_start = matColStart[sec];
        while(sec < matRows[0] && sec_end - sec_start > row) {
            // if(row == 3) printf("%d\n",sec);
            dot += matData[matColStart[sec] + row] * vec[matCols[matColStart[sec] + row]];
            sec++;
            sec_end = sec == (matRows[0] - 1) ? ndata : matColStart[sec + 1];
            sec_start = matColStart[sec];
            // if(row == 3) printf("%d %d\n", sec, sec_end - sec_start);
        }
        out[matRowPerm[row]] = dot;
        // printf("matRowPerm[%d] = %d, dot = %f \n", row, matRowPerm[row], dot);
    }

}

// static void spmvJDS(float *out, int *matColStart, int *matCols,
//                     int *matRowPerm, int *matRows, float *matData,
//                     float *vec, int dim) {

//     //@@ invoke spmv kernel for jds format
    
// }

int main(int argc, char **argv)
{
    wbArg_t args;
    int *hostCSRCols;
    int *hostCSRRows;
    float *hostCSRData;
    int *hostJDSColStart;
    int *hostJDSCols;
    int *hostJDSRowPerm;
    int *hostJDSRows;
    float *hostJDSData;
    float *hostVector;
    float *hostOutput;
    int *deviceJDSColStart;
    int *deviceJDSCols;
    int *deviceJDSRowPerm;
    int *deviceJDSRows;
    float *deviceJDSData;
    float *deviceVector;
    float *deviceOutput;
    int dim, ncols, nrows, ndata;
    int maxRowNNZ;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostCSRCols = (int *)wbImport(wbArg_getInputFile(args, 0), &ncols, "Integer");
    hostCSRRows = (int *)wbImport(wbArg_getInputFile(args, 1), &nrows, "Integer");
    hostCSRData = (float *)wbImport(wbArg_getInputFile(args, 2), &ndata, "Real");
    hostVector = (float *)wbImport(wbArg_getInputFile(args, 3), &dim, "Real");

    hostOutput = (float *)malloc(sizeof(float) * dim);

    wbTime_stop(Generic, "Importing data and creating memory on host");

    CSRToJDS(dim, hostCSRRows, hostCSRCols, hostCSRData, &hostJDSRowPerm, &hostJDSRows,
             &hostJDSColStart, &hostJDSCols, &hostJDSData);
    maxRowNNZ = hostJDSRows[0];

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **)&deviceJDSColStart, sizeof(int) * maxRowNNZ);
    cudaMalloc((void **)&deviceJDSCols, sizeof(int) * ndata);
    cudaMalloc((void **)&deviceJDSRowPerm, sizeof(int) * dim);
    cudaMalloc((void **)&deviceJDSRows, sizeof(int) * dim);
    cudaMalloc((void **)&deviceJDSData, sizeof(float) * ndata);

    cudaMalloc((void **)&deviceVector, sizeof(float) * dim);
    cudaMalloc((void **)&deviceOutput, sizeof(float) * dim);
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceJDSColStart, hostJDSColStart, sizeof(int) * maxRowNNZ,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSCols, hostJDSCols, sizeof(int) * ndata, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSRowPerm, hostJDSRowPerm, sizeof(int) * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSRows, hostJDSRows, sizeof(int) * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSData, hostJDSData, sizeof(float) * ndata, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVector, hostVector, sizeof(float) * dim, cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");

    dim3 dimGrid(ceil((float)dim / BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    
    spmvJDSKernel<<<dimGrid, dimBlock>>>(deviceOutput, deviceJDSColStart, deviceJDSCols, deviceJDSRowPerm, deviceJDSRows,
            deviceJDSData, deviceVector, dim, ndata);
    

    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * dim, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceVector);
    cudaFree(deviceOutput);
    cudaFree(deviceJDSColStart);
    cudaFree(deviceJDSCols);
    cudaFree(deviceJDSRowPerm);
    cudaFree(deviceJDSRows);
    cudaFree(deviceJDSData);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, dim);

    free(hostCSRCols);
    free(hostCSRRows);
    free(hostCSRData);
    free(hostVector);
    free(hostOutput);
    free(hostJDSColStart);
    free(hostJDSCols);
    free(hostJDSRowPerm);
    free(hostJDSRows);
    free(hostJDSData);

    return 0;
}
