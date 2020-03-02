#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double dot_product(double* restrict a, unsigned int k, unsigned int j){
        int m = 0, TAM2 = 2209;
        double sum = 0;
        #pragma unroll
        for (unsigned int m = 0; m < TAM2; m += 47) {
            sum += a[m + k] * a[m + j];
        }
        return sum;
}

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void qr_method(global double* restrict dataQ, global double* restrict dataR){
    const unsigned int TAM = 47, TAM2 = 2209;
    double r2[TAM], rn[TAM2];
    double localQ[TAM2], localR[TAM2];

    for (unsigned int i = 0; i < TAM2; i++){
        localQ[i] = dataQ[i];
        localR[i] = 0;
    }

    unsigned int k_stride = 0;
    for (unsigned int k = 0; k < TAM; k++) {
	r2[k] = dot_product(localQ, k, k);

        for (unsigned int j = k + 1; j < TAM; j++) {
            rn[k_stride + j] = dot_product(localQ, k, j);
        }

        for (unsigned int j = k + 1; j < TAM; j++) {
	    double rn_divided_by_r2 = rn[k_stride + j] / r2[k];
	    double Q_column_buffer[TAM];

	    #pragma unroll
	    for (unsigned int m = 0, i = 0; m < TAM2; m += 47, i++) {
               Q_column_buffer[i] = localQ[m + j] - rn_divided_by_r2 * localQ[m + k];
            }

            for (unsigned int m = 0, i = 0; m < TAM2; m += 47, i++) {
                localQ[m + j] = Q_column_buffer[i];
            }
        }
        k_stride += 47;
    }

    unsigned int k_stride_2 = 0;
    for (unsigned int k = 0; k < TAM; k++) {
        localR[k_stride_2 + k] = sqrt(r2[k]);
        double rkk = localR[k_stride_2 + k];

        for (unsigned int j = k+1; j < TAM; j++) {
            localR[k_stride_2 + j] = rn[k_stride_2 + j]/rkk;
        }

        for (unsigned int m = 0; m < TAM2; m += 47) {
            localQ[m + k] = localQ[m + k]/rkk;
        }

        k_stride_2 += 47;
    }

    for (unsigned int i = 0; i < TAM2; i++){
        dataQ[i] = localQ[i];
        dataR[i] = localR[i];
    }
}
