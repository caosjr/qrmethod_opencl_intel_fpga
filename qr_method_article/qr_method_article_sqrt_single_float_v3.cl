float dot_product(float* restrict a, unsigned int k, unsigned int j){
        int m = 0, TAM2 = 2209;
        float sum = 0;
        #pragma unroll
        for (m = 0; m < TAM2; m += 47) {
            sum += a[m + k] * a[m + j];
        }
        return sum;
}

__kernel
__attribute__((reqd_work_group_size(1,1,1)))
void qr_method(global float* restrict dataQ, global float* restrict dataR){

    unsigned int i, j, k, m, k_stride;
    const unsigned int TAM=47, TAM2 = 2209;
    float sum, sum2;
    float r2[TAM];
    float rn[TAM2];
    float localQ[TAM2], localR[TAM2], Q_column_buffer[TAM];

    for (i = 0; i < TAM2; i++){
        localQ[i] = dataQ[i];
        localR[i] = 0;
    }

    k_stride = 0;
    for (k = 0; k < TAM; k++) {
	r2[k] = dot_product(localQ, k, k);

        for (j = k + 1; j < TAM; j++) {
            rn[k_stride + j] = dot_product(localQ, k, j);
        }

        for (j = k + 1; j < TAM; j++) {
	    i = 0;
            for (m = 0; m < TAM2; m += 47) {
                Q_column_buffer[i] = localQ[m + j] - ((rn[k_stride + j] / r2[k]) * localQ[m + k]);
		i++;
            }
	    i = 0;
	    for (m = 0; m < TAM2; m += 47) {
                localQ[m + j] = Q_column_buffer[i];
		i++;
            }
        }
        k_stride += 47;
    }

    k_stride = 0;
    for (k = 0; k < TAM; k++) {
        localR[k_stride + k] = sqrt(r2[k]);
        float rkk = localR[k_stride + k];

        for (j = k+1; j < TAM; j++) {
            localR[k_stride + j] = rn[k_stride + j]/rkk;
        }

        for (m = 0; m < TAM2; m += 47) {
            localQ[m + k] = localQ[m + k]/rkk;
        }

        k_stride += 47;
    }

    for (i = 0; i < TAM2; i++){
        dataQ[i] = localQ[i];
        dataR[i] = localR[i];
    }
}
