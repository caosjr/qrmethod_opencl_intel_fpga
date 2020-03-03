#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void qr_method(__global double* restrict dataQ, __global double* restrict dataR){

    unsigned int i, j, k, m, matrix_id, block_id;
    const unsigned int TAM = 47, TAM2 = 2209, matrix_block_size = 28717; //cinco blocos com treze matrizes
    double sum, rkk, dataQ_line[TAM], localQ[matrix_block_size], localR[matrix_block_size];

    for (block_id = 0; block_id < 5; block_id++) {
        for (matrix_id = 0; matrix_id < matrix_block_size; matrix_id++) {
            localQ[matrix_id] = dataQ[matrix_id];
            localR[matrix_id] = dataR[matrix_id];
        }

        for (matrix_id = 0; matrix_id < matrix_block_size; matrix_id = matrix_id + TAM2) {
            for (k = 0; k < TAM; k++) {
                sum = 0;
                #pragma unroll
                for (m = 0; m < TAM2; m += 47) {
                    sum += localQ[m + k + matrix_id] * localQ[m + k + matrix_id];
                }
                localR[k * TAM + k] = sqrt(sum); //pow(sum, onehalf);//sqrt(sum);

                for (m = 0; m < TAM2; m += 47) {
                    localQ[m + k + matrix_id] = localQ[m + k + matrix_id] * localR[k * TAM + k + matrix_id];
                }

                for (j = k + 1; j < TAM; j++) {
                    sum = 0;
                    #pragma unroll
                    for (m = 0; m < TAM2; m += 47) {
                        sum += localQ[m + j + matrix_id] * localQ[m + k + matrix_id];
                    }
                    localR[k * TAM + j + matrix_id] = sum;

                    i = 0;
                    for (m = 0; m < TAM2; m += 47) {
                        dataQ_line[i] = localQ[m + j + matrix_id] - (localR[k * TAM + j + matrix_id] * localQ[m + k + matrix_id]);
                        i++;
                    }

                    i = 0;
                    for (m = 0; m < TAM2; m += 47) {
                        localQ[m + j + matrix_id] = dataQ_line[i];
                        i++;
                    }
                }
            }
        }

        for (matrix_id = 0; matrix_id < matrix_block_size; matrix_id++) {
            dataQ[matrix_id] = localQ[matrix_id];
            dataR[matrix_id] = localR[matrix_id];
        }
    }
}

