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
    void update_intermediates();

    //amplitudes
    double t1(int a, int i);
    double t2(int a, int b, int i, int j);
    double t3(int a, int b, int c, int i, int j, int k);

    mat t1a;
    //field<mat> t2a;
    SpMat<double> t2a;
    SpMat<double> t3a;
    //t3amps t3a;
    //SpMat<double> t3a;
    //sp_mat t3a;



    //parameters
    int iNs; //number of states
    int iNs2; // squared number of particles
    int iNp; //number of particles






};

#endif // CCSOLVE_H
