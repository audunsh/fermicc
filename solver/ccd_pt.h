#ifndef CCD_PT_H
#define CCD_PT_H

#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>
#include "basis/electrongas.h"
#include "solver/flexmat.h"
#include "solver/flexmat6.h"
#include "solver/initializer.h"
#include "solver/unpack_sp_mat.h"

using namespace std;
using namespace arma;


class ccd_pt
{
public:
    //CCD with perturbative triplets
    ccd_pt(electrongas bs, double a);
    double alpha;

    electrongas ebs;
    initializer iSetup;

    //flexmat Vhh();
    flexmat vhhhh;
    flexmat vpppp;
    flexmat vhhpp;
    flexmat vpphh;
    flexmat vhpph;
    flexmat vhphh;
    flexmat vppph;
    flexmat vhppp;



    flexmat vphpp;
    flexmat vhhhp;




    flexmat T;
    flexmat Tprev;
    //flexmat Vpppp();
    //flexmat Vhhpp();

    //flexmat Vpphh();
    //flexmat Vhpph();
    //flexmat T();
    void advance();
    void L1_dense_multiplication(); //block ladder calculation

    void check_matrix_consistency();

    void energy();

    //The diagrams contributing to the CCD energy
    sp_mat L1, L2, L3, Q1, Q2, Q3, Q4;
    flexmat fmL3, fmQ1, fmQ2, fmQ3, fmQ4;

    flexmat fmI1, fmI2, fmI3, fmI4, fmI2temp, fmI3temp; //intermediates for CCD terms

    flexmat fmD10b, fmD10c;


    flexmat6 T3, t2a, t2b;

    double correlation_energy;

    int iterations;


};

#endif // CCD_PT_H
