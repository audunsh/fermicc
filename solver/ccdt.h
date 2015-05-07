#ifndef CCDT_H
#define CCDT_H

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


class ccdt
{
public:
    ccdt(electrongas bs);


    electrongas ebs;
    initializer iSetup;

    //flexmat Vhh();
    flexmat vhhhh;
    flexmat vpppp;
    flexmat vhhpp;
    flexmat vpphh;
    flexmat vhpph;

    //triples specific interactions
    flexmat vphhp;
    flexmat vppph;
    flexmat vhppp;
    flexmat vhphh;
    flexmat vhhhp;
    flexmat vphpp;
    flexmat vhhph;

    //amplitudes
    flexmat T;
    flexmat6 T3;


    void advance();
    void advance_intermediates(); //advance using intermediates
    void L1_block_multiplication(); //block ladder calculation
    void L1_dense_multiplication(); //block ladder calculation

    void check_matrix_consistency();

    void energy();

    //The diagrams contributing to the CCD energy
    sp_mat L1, L2, L3, Q1, Q2, Q3, Q4;
    flexmat fmL3, fmQ1, fmQ2, fmQ3, fmQ4, fmD10b, fmD10c;
    flexmat6 t2t3a, t2t3b,t2t3c,t2t3d,t2t3e,t2t3f,t2t3g, t2t2b,t2t2c,t2t2d, t3b,t3c,t2a,t2b;


    flexmat fmI1, fmI2, fmI3, fmI4, fmI2temp, fmI3temp; //intermediates for CCD terms

    double CCSD_SG_energy();
    double correlation_energy;


};

#endif // CCDT_H
