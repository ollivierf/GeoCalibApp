#include <stdio.h>
#include <stdlib.h>


double norm_1d_c(double *vect, int vect_size){
    double val = 0.0;
    int idx = 0;

    for(idx=0; idx<vect_size; idx++){
        val += vect[idx]*vect[idx];
    }
    return sqrt(val);
}


int MakeDsr_c(double *Dsr, double *s, double *r, int K, int L){

    int k, l, j = 0;
    double smr[3];

    for(k=0; k<K; k++){
        for(l=0; l<L; l++){
            for(j=0; j<3; j++){
                smr[j] = s[k*3 + j] - r[l*3 + j];
            }
            Dsr[k*L + l] = norm_1d_c(smr, 3);
        }
    }
    return 1;
}

int UpdateEpsMumMun_c(double *eps, double *mum, double *mun,
                      double *Dsr, double *tau, double *t, double *o,
                      double c0, int K, int L){

    int k, m, n = 0;

    for(k=0; k<K; k++){
        for(m=0; m<L; m++){
            for(n=0; n<L; n++){
                eps[k*L*L + m*L + n] = Dsr[k*L + m] - Dsr[k*L + n] - c0*(tau[k*L*L + m*L + n] - o[k*L*L + m*L + n] + t[m] - t[n]);
                mum[k*L*L + m*L + n] = Dsr[k*L + m] - 0.5*eps[k*L*L + m*L + n];
                mun[k*L*L + m*L + n] = Dsr[k*L + n] + 0.5*eps[k*L*L + m*L + n];
            }
        }
    }
    return 1;
}


int UpdateE_c(double *e_, double *s, double *r, double *Dsr, int K, int L){

    int k, n, j = 0;

    for(k=0; k<K; k++){
            for(n=0; n<L; n++){
                for(j=0; j<3; j++){
                    e_[k*L*3 + n*3 + j] = (s[k*3 + j] - r[n*3 + j]) / Dsr[k*L + n];
            }
        }
    }
    return 1;
}

int UpdateS_c(double *mum, double *e_, double *s, double *r, int K, int L){

    int i, m, j = 0;
    double tot[3];
    double sum_mum_im;

    for(i=0; i<K; i++){
        for(j=0; j<3; j++){tot[j] = 0.0;}
        for(m=0; m<L; m++){
            sum_mum_im = 0.0;
            for(j=0; j<L; j++){
                sum_mum_im += mum[i*L*L + m*L + j];
            }
            for(j=0; j<3; j++){
                tot[j] += (double)L*r[m*3 + j] + e_[i*L*3 + m*3 + j]*sum_mum_im;
            }
        }
        for(j=0; j<3; j++){
            s[i*3 + j] = tot[j] /L/L;
        }
    }
    return 1;
}

int UpdateR_c(double *mun, double *e_, double *r, double *s, int K, int L){

    int i, n, j = 0;
    double tot[3];
    double sum_mun_in;

    for(n=0; n<L; n++){
        for(j=0; j<3; j++){tot[j] = 0.0;} //initialize tot to 0s
        for(i=0; i<K; i++){

            //compute sum_mum_im
            sum_mun_in = 0.0;
            for(j=0; j<L; j++){
                sum_mun_in += mun[i*L*L + j*L + n];
            }

            //compute tot 
            for(j=0; j<3; j++){
                tot[j] += (double)L*s[i*3 + j] - e_[i*L*3 + n*3 + j]*sum_mun_in;
            }
        }
        // update r
        for(j=0; j<3; j++){
            r[n*3 + j] = tot[j] /L/K;
        }
    }
    return 1;
}



int UpdateT_c(double *t, double *Dsr, double *mun, double c0, int K, int L){

    int n, i, j;
    double tot;
    double sum_mun_in;

    for(n=0; n<L; n++){
        tot = 0.0;
        for(i=0; i<K; i++){

            //compute sum_mum_im
            sum_mun_in = 0.0;
            for(j=0; j<L; j++){
                sum_mun_in += mun[i*L*L + j*L + n];
            }

            tot += (double)L*Dsr[i*L + n] - sum_mun_in;
        }
        // update t
        t[n] = tot/L/K/c0;
    }
    return 1;
}

int Step_c(double *Dsr, double *eps, double *mum, double *mun,
           double *e_, double *s, double *r, double *tau, double *t, double *o,
           double c0, int K, int L){

    MakeDsr_c(Dsr, s, r, K, L);
    UpdateEpsMumMun_c(eps, mum, mun, Dsr, tau, t, o, c0, K, L);
    UpdateE_c(e_, s, r, Dsr, K, L);
    UpdateS_c(mum, e_, s, r, K, L);
    UpdateR_c(mun, e_, r, s, K, L);
    //UpdateT_c(t, Dsr, mun, c0, K, L);

    return 1;
}

