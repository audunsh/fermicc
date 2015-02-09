#ifndef CCSOLVE_H
#define CCSOLVE_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>
#include "basis/electrongas.h"
//#include "solver/t3amps.h"

using namespace std;
using namespace arma;

class ccsolve
{
public:
    ccsolve();
    ccsolve(electrongas f);
    double v(int a, int b, int i, int j); //two-particle interaction
    double h(int a, int i);               //single particle energies
    electrongas eBasis;

    //solver functions
    void initialize_amplitudes();
    void initialize_t3amplitudes();
    void update_intermediates();

    //Stanton-Gauss intermediates
    double CCSD_SG(int iNparticles);

    double CCSD_SG_dt1(int a, int i);
    double CCSD_SG_dt2(int a, int b, int i, int j);

    void update_SGIntermediates();
    void initialize_SGIntermediates();
    field<mat> w1a; //4D-tensors
    field<mat> w2a;
    field<mat> w3a;
    field<mat> w4a;
    mat f1a;        //2D-tensors
    mat f2a;
    mat f3a;

    //amplitudes
    double t1(int a, int i);
    double t2(int a, int b, int i, int j);
    double t3(int a, int b, int c, int i, int j, int k);

    mat t1a;
    SpMat<double> t2a;
    SpMat<double> t3a;

    //parameters
    int iNs; //number of states
    int iNs2; // squared number of particles
    int iNp; //number of particles

    //calibration functions
    void scan_amplitudes(); //scan through all elements in the amplitudes

};

#endif // CCSOLVE_H
