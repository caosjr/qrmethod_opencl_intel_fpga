#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void qr_method(__global double* restrict dataQ, __global double* restrict dataR){

    unsigned int i, j, k, m;
    const unsigned int TAM = 47, TAM2 = 2209;
    double sum, rkk, sum_pipeline[TAM], dataQ_line[TAM], localQ[TAM2], localR[TAM2];
    //fast inverse square root
    long long inverse;
    double x2, y, y_final;
    const double threehalfs = 1.5;

    for (i = 0; i < TAM2; i++){
        localQ[i] = dataQ[i];
        localR[i] = dataR[i];
    }

    for (k = 0; k < TAM; k++){
        sum = 0;
        i = 0;
        for (m = 0; m < TAM2; m += 47){
            sum_pipeline[i] = localQ[m + k] * localQ[m + k];
            i++;
        }

        #pragma unroll
        for (m = 0; m < TAM; m++){
            sum += sum_pipeline[m];
        }
        x2 = sum * 0.5;
        y = sum;
        inverse  = * (long long *) &y;
        inverse  = 0x5FE6EB50C7B537A9 - (inverse >> 1);
        y  = * (double*) &inverse;
        y_final  = y * (threehalfs - (x2 * y * y));

        localR[k*TAM + k] = y_final; //pow(sum, onehalf);//sqrt(sum);

        for (m = 0; m < TAM2; m += 47) {
            localQ[m + k] = localQ[m + k]*localR[k*TAM + k];
        }

        for (j = k+1; j < TAM; j++){
            sum = 0;
            i = 0;
            for (m = 0; m < TAM2; m += 47){
                sum_pipeline[i] = localQ[m+ j] * localQ[m + k];
                i++;
            }

            #pragma unroll
            for (m = 0; m < TAM; m++){
                sum += sum_pipeline[m];
            }
            localR[k*TAM + j] = sum;
            i = 0;

            for (m = 0; m < TAM2; m += 47){
                dataQ_line[i] = localQ[m + j] - (localR[k*TAM + j] * localQ[m + k]);
                i++;
            }

            i = 0;
            for (m = 0; m < TAM2; m += 47){
                localQ[m + j] = dataQ_line[i];
                i++;
            }
        }
    }

    for (i = 0; i < TAM2; i++){
        dataQ[i] = localQ[i];
        dataR[i] = localR[i];
    }
}

