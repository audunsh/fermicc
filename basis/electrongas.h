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
    void generate_state_list(int Ne, double rs);
    double absdiff2(rowvec A, rowvec B);
    double v(int P, int Q, int R, int S);
    double f(int P, int Q);
    double h(int P, int Q);
    double eref(int nParticles);
    double analytic_energy(int nParticles);
    int kd_vec(rowvec A, rowvec B);
    int kd(int A, int B);

    double dL;
    double dL3; //L*L*L
    double dr_s;
    int iN;
    int iNbstates;

    mat mSortedEnergy;
    vec vEnergy;
    double dPrefactor1;
    double dPrefactor2;

    double pi = 4*atan(1);
};

#endif // ELECTRONGAS_H
