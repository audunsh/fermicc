#ifndef ELECTRONGAS_H
#define ELECTRONGAS_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>

using namespace std;
using namespace arma;

class electrongas
{
public:
    electrongas();
    void generate_state_list2(int Ne, double rs, u64 Np);
    void generate_state_list(int Ne, double rs, u64 Np);
    vec F_vectorized(uint p); //fock eigenvalues vectorized
    double absdiff2(rowvec A, rowvec B);
    double v(int P, int Q, int R, int S);
    double v2(int p, int q, int r, int s);
    double v3(int p, int q, int r, int s); //skip momentum kroenecker k_p + k_q == k_r + k_s
    double v4(int p, int q); //for the case <pq||pq>
    ivec unique(uvec p);
    int unique_int(uint p);

    void print();


    double f(int P, int Q);
    double F(int p);


    double h(int P, int Q);
    double eref(int nParticles);
    double analytic_energy(int nParticles);
    int kd_vec(rowvec A, rowvec B);
    int kd(int A, int B);


    double dL;
    double dL3; //L*L*L
    double dL2; //L*L
    double dr_s;
    u64 iN;
    u64 iNbstates;

    mat mSortedEnergy;
    vec vEnergy;
    vec vHFEnergy;
    double dPrefactor1;
    double dPrefactor2;
    double mu;
    u64 iNparticles;
    int k_step;

    double pi = 4*atan(1);
    //double pi = 3.1415;

    //added vectorized functionality, 21.03.2015
    ivec vKx,vKy,vKz, vMs;

};

#endif // ELECTRONGAS_H
