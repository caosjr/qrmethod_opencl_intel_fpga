#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void qrmethod(__global double *restrict dataQ, __global double * restrict dataR){

    unsigned int j, k, m;
    const unsigned int TAM=47;
    double sum, rkk, sum_pipeline[TAM], dataQ_line[TAM];
    //fast inverse square root
    long long i;
    double x2, y, y_final;
    const double threehalfs = 1.5;

    for (k = 0; k < TAM; k++){
        sum = 0;
        for (m = 0; m < TAM; m++){
			sum_pipeline[m] = dataQ[m*TAM + k] * dataQ[m*TAM + k];
        }

        #pragma unroll
        for (m = 0; m < TAM; m++){
            sum += sum_pipeline[m];
        }
        x2 = sum * 0.5;
        y = sum;
        i  = * (long long *) &y;
        i  = 0x5FE6EB50C7B537A9 - (i >> 1);
        y  = * (double *) &i;
        y_final  = y * (threehalfs - (x2 * y * y));

        dataR[k*TAM + k] = y_final; //pow(sum, onehalf);//sqrt(sum);
        
        for (m = 0; m < TAM; m ++) {
            sum_pipeline[m] = dataQ[m*TAM + k]*dataR[k*TAM + k];
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

