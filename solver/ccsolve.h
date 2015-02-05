#ifndef CCSOLVE_H
#define CCSOLVE_H

#include "basis/electrongas.h"

#include <armadillo>
#include <iomanip>

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

    sp_mat md_t1amps;
    field<sp_mat> fm_t2amps; //(field<mat>);
    field<sp_mat> fm_gas;

    //parameters
    int iNs; //number of states
    int iNp; //number of particles






};

#endif // CCSOLVE_H
