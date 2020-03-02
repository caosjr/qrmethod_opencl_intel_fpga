#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void qrmethod(__global double* restrict dataQ, __global double* restrict dataR){

    unsigned int i, j, k, m, matrix_id;
    const unsigned int TAM = 47, TAM2 = 2209, matrix_block_size = 143585;
    double sum, rkk, sum_pipeline[TAM], dataQ_line[TAM], localQ[TAM2], localR[TAM2];

    for (matrix_id = 0; matrix_id < matrix_block_size; matrix_id = matrix_id + TAM2) {
        //copia de uma parte do bloco dentro da memoria local
        for (i = 0; i < TAM2; i++) {
            localQ[i] = dataQ[i + matrix_id];
            localR[i] = dataR[i + matrix_id];
        }

        for (k = 0; k < TAM; k++) {
            sum = 0;
            i = 0;
            for (m = 0; m < TAM2; m += 47) {
                sum_pipeline[i] = localQ[m + k] * localQ[m + k];
                i++;
            }

            #pragma unroll
            for (m = 0; m < TAM; m++) {
                sum += sum_pipeline[m];
            }

            localR[k * TAM + k] = sqrt(sum);

            for (m = 0; m < TAM2; m += 47) {
                localQ[m + k] = localQ[m + k] / localR[k * TAM + k];
            }

            for (j = k + 1; j < TAM; j++) {
                sum = 0;
                i = 0;
                for (m = 0; m < TAM2; m += 47) {
                    sum_pipeline[i] = localQ[m + j] * localQ[m + k];
                    i++;
                }

                #pragma unroll
                for (m = 0; m < TAM; m++) {
                    sum += sum_pipeline[m];
                }
                localR[k * TAM + j] = sum;
                i = 0;

                for (m = 0; m < TAM2; m += 47) {
                    dataQ_line[i] = localQ[m + j] - (localR[k * TAM + j] * localQ[m + k]);
                    i++;
                }

                i = 0;
                for (m = 0; m < TAM2; m += 47) {
                    localQ[m + j] = dataQ_line[i];
                    i++;
                }
            }
        }

        for (i = 0; i < TAM2; i++) {
            dataQ[i] = localQ[i];
            dataR[i] = localR[i];
        }
    }
}
