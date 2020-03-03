#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void qr_method(__global double *restrict dataQ, __global double * restrict dataR){

    unsigned int j, k, m;
    const unsigned int TAM=47;
    double sum, rkk;
    //fast inverse square root
    long long i;
    double x2, y;
    const double threehalfs = 1.5;

    for (k = 0; k < TAM; k++){
        sum = 0;
        for (m = 0; m < TAM; m++){
			sum += dataQ[m*TAM + k] * dataQ[m*TAM + k];
        }
        x2 = sum * 0.5;
        y = sum;
        i  = * (long long *) &y;
        i  = 0x5FE6EB50C7B537A9 - (i >> 1);
        y  = * (double *) &i;
        y  = y * (threehalfs - (x2 * y * y));
        
        dataR[k*TAM + k] = y; //pow(sum, onehalf);//native_sqrt(sum);

        for (m = 0; m < TAM; m++) {
            //dataQ[m*TAM + k] = dataQ[m*TAM + k]/dataR[k*TAM + k];
            dataQ[m*TAM + k] = dataQ[m*TAM + k]*dataR[k*TAM + k];
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

