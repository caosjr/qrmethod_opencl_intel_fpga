#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void qr_method(global double* restrict dataQ, global double* restrict dataR){

    unsigned int i, j, k, m, k_stride;
    const unsigned int TAM=47, TAM2 = 2209;
    double sum, rkk, sum_pipeline[TAM];
    double r2[TAM];
    double rn[TAM2];
    double localQ[TAM2], localR[TAM2], localQ_line[TAM2];

    #pragma unroll 3
    for (i = 0; i < TAM2; i++){
        localQ[i] = dataQ[i];
        localR[i] = 0;
    }

    k_stride = 0;
    for (k = 0; k < TAM; k++) {
        sum = 0;
        i = 0;
        for (m = 0; m < TAM2; m += 47) {
            sum_pipeline[i] = localQ[m + k] * localQ[m + k];
            i++;
        }

        #pragma unroll
        for (i = 0; i < TAM; i++){
            sum += sum_pipeline[i];
        }

        r2[k] = sum;
        for (j = k + 1; j < TAM; j++) {
            sum = 0;
            i = 0;
            for (m = 0; m < TAM2; m += 47) {
                sum_pipeline[i] = localQ[m + k] * localQ[m + j];
                i++;
            }
            #pragma unroll
            for (i = 0; i < TAM; i++){
                sum += sum_pipeline[i];
            }
            rn[k_stride + j] = sum;
        }

        for (j = k + 1; j < TAM; j++) {
            i = 0;
            for (m = 0; m < TAM2; m += 47) {
                localQ_line[i] = localQ[m + j] - ((rn[k_stride + j] / r2[k]) * localQ[m + k]);
                i++;
            }
            i = 0;
            for (m = 0; m < TAM2; m += 47) {
                localQ[m + j] = localQ_line[i];
                i++;
            }
        }
        k_stride += 47;
    }

    k_stride = 0;
    for (k = 0; k < TAM; k++) {
        localR[k_stride + k] = sqrt(r2[k]);
        rkk = localR[k_stride + k];

        for (j = k+1; j < TAM; j++) {
            localR[k_stride + j] = rn[k_stride + j]/rkk;
        }

        i = 0;
        for (m = 0; m < TAM2; m += 47) {
            sum_pipeline[i] = localQ[m + k]/rkk;
            i++;
        }

        i = 0;
        #pragma unroll
        for (m = 0; m < TAM2; m += 47) {
            localQ[m + k] = sum_pipeline[i];
            i++;
        }
        k_stride += 47;
    }

    #pragma unroll 2
    for (i = 0; i < TAM2; i++){
        dataQ[i] = localQ[i];
        dataR[i] = localR[i];
    }
}

