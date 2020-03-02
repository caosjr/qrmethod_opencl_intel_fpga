#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void qrmethod(__global double *restrict dataQ, __global double * restrict dataR){

    unsigned int j, k, m;
    const unsigned int TAM=47;
    const double onehalf = 1/2;
    double sum, rkk, sum_pipeline[TAM], dataQ_line[TAM];

    for (k = 0; k < TAM; k++){
        sum = 0;
        for (m = 0; m < TAM; m++){
			sum_pipeline[m] = dataQ[m*TAM + k] * dataQ[m*TAM + k];
        }

        #pragma unroll
        for (m = 0; m < TAM; m++){
            sum += sum_pipeline[m];
        }

        dataR[k*TAM + k] = pow(sum, onehalf);
        
        for (m = 0; m < TAM; m ++) {
            sum_pipeline[m] = dataQ[m*TAM + k]/dataR[k*TAM + k];
        }

        for (m = 0; m < TAM; m ++) {
            dataQ[m*TAM + k] = sum_pipeline[m];
        }

        for (j = k+1; j < TAM; j++){
            sum = 0;
            for (m = 0; m < TAM; m++){
                sum_pipeline[m] = dataQ[m*TAM + j] * dataQ[m*TAM + k];
            }

            #pragma unroll
            for (m = 0; m < TAM; m++){
                sum += sum_pipeline[m];
            }
            dataR[k*TAM + j] = sum;

            for (m = 0; m < TAM; m++){
                dataQ_line[m] = dataQ[m*TAM + j] - (dataR[k*TAM + j] * dataQ[m*TAM + k]);
            }

            for (m = 0; m < TAM; m++){
                dataQ[m*TAM + j] = dataQ_line[m];
            }
        }
    }
}

