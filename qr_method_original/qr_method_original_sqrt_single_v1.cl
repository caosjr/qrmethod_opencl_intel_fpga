#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void qrmethod(__global double *restrict dataQ, __global double * restrict dataR){

    unsigned int j, k, m;
    const unsigned int TAM=47;
    double sum, rkk;

    for (k = 0; k < TAM; k++){
        sum = 0;
        for (m = 0; m < TAM; m++){
			sum += dataQ[m*TAM + k] * dataQ[m*TAM + k];
        }

        dataR[k*TAM + k] = sqrt(sum);

        for (m = 0; m < TAM; m++) {
            dataQ[m*TAM + k] = dataQ[m*TAM + k]/dataR[k*TAM + k];
        }

        for (j = k+1; j < TAM; j++){
            sum = 0;
            for (m = 0; m < TAM; m++){
                sum += dataQ[m*TAM + j] * dataQ[m*TAM + k];
            }
            dataR[k*TAM + j] = sum;

            for (m = 0; m < TAM; m++){
                dataQ[m*TAM + j] = dataQ[m*TAM + j] - (dataR[k*TAM + j] * dataQ[m*TAM + k]);
            }
        }
    }
}

