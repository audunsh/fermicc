#include "ccd.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "solver/flexmat.h"
#include "basis/electrongas.h"
#include "solver/initializer.h"
#include "solver/unpack_sp_mat.h"

using namespace std;
using namespace arma;

ccd::ccd(electrongas bs){
    ebs = bs;
    iSetup = initializer(bs);

    //setup all interaction matrices
    iSetup.sVhhhh();
    iSetup.sVpppp();
    iSetup.sVhhpp();
    iSetup.sVpphh();
    iSetup.sVhpph();

    //convert interaction data to flexmat objects
    vhhhh.init(iSetup.vValsVhhhh, iSetup.iVhhhh, iSetup.jVhhhh, iSetup.kVhhhh, iSetup.lVhhhh, iSetup.iNh, iSetup.iNh, iSetup.iNh, iSetup.iNh);
    vpppp.init(iSetup.vValsVpppp, iSetup.aVpppp, iSetup.bVpppp, iSetup.cVpppp, iSetup.dVpppp, iSetup.iNp, iSetup.iNp, iSetup.iNp, iSetup.iNp);
    vhpph.init(iSetup.vValsVhpph, iSetup.iVhpph, iSetup.aVhpph, iSetup.bVhpph, iSetup.jVhpph, iSetup.iNh, iSetup.iNp, iSetup.iNp, iSetup.iNh);
    vhhpp.init(iSetup.vValsVhhpp, iSetup.iVhhpp, iSetup.jVhhpp, iSetup.aVhhpp, iSetup.bVhhpp, iSetup.iNh, iSetup.iNh, iSetup.iNp, iSetup.iNp);

    //set up first T2-amplitudes
    T.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    T.set_amplitudes(bs.vEnergy);


    // HOW TO SET UP FLEXMAT OBJECTS FROM CSC-MATRICES
    // flexmat V1;
    // V1.update(vhhhh.pq_rs(),vhhhh.iNp, vhhhh.iNq, vhhhh.iNr, vhhhh.iNs); //update (or initialize) with an sp_mat object (requires unpacking)


    energy();
}

void ccd::advance(){
    //advance the solution one step
    L1 = vpppp.pq_rs()*T.pq_rs();
    L2 = T.pq_rs()*vhhhh.pq_rs();
    L3 = vhpph.sq_rp()*T.qs_pr(); //needs realignment and permutations
    Q1 = T.pq_rs()*vhhpp.pq_rs()*T.pq_rs();
    Q2 = T.pr_sq()*vhhpp.pr_qs()*T.rq_ps(); //needs realignment and permutations
    Q3 = T.pqs_r()*vhhpp.q_prs()*T.sqp_r(); //needs realignment and permutations
    Q4 = T.p_srq()*vhhpp.pqr_s()*T.p_qrs(); //needs realignment and permutations


    //Q2
    //Q3
    //Q4
    T.update(.5*L1 + .5*L2 + L3 + .25*Q1 + Q2 - .5*Q3 - .5*Q4, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    T.set_amplitudes(ebs.vEnergy);
}

void ccd::energy(){
    //Calculate the ground state energy
    sp_mat cv = vhhpp.pq_rs() * T.pq_rs();
    mat Cv(cv); //this is inefficient: does not utilize sp_mat functionality, one possibility through unpack_sp_mat
    double C_ = 0;
    for(int i = 0; i<Cv.n_cols; i++){
        C_+= Cv(i,i);
    }
    cout << .25*C_ << endl;

}
