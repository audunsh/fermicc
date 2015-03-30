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
    //flexmat Vhh(iSetup.vValsVhhhh, iSetup.iVhhhh, iSetup.jVhhhh, iSetup.kVhhhh, iSetup.lVhhhh, iSetup.iNh, iSetup.iNh, iSetup.iNh, iSetup.iNh);

    //convert interaction data to flexmat objects
    vhhhh.init(iSetup.vValsVhhhh, iSetup.iVhhhh, iSetup.jVhhhh, iSetup.kVhhhh, iSetup.lVhhhh, iSetup.iNh, iSetup.iNh, iSetup.iNh, iSetup.iNh);
    vpppp.init(iSetup.vValsVpppp, iSetup.aVpppp, iSetup.bVpppp, iSetup.cVpppp, iSetup.dVpppp, iSetup.iNp, iSetup.iNp, iSetup.iNp, iSetup.iNp);
    vhpph.init(iSetup.vValsVhpph, iSetup.iVhpph, iSetup.aVhpph, iSetup.bVhpph, iSetup.jVhpph, iSetup.iNh, iSetup.iNp, iSetup.iNp, iSetup.iNh);
    vhhpp.init(iSetup.vValsVhhpp, iSetup.iVhhpp, iSetup.jVhhpp, iSetup.aVhhpp, iSetup.bVhhpp, iSetup.iNh, iSetup.iNh, iSetup.iNp, iSetup.iNp);
    T.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    T.set_amplitudes(bs.vEnergy);
    //unpack_sp_mat H(T.pr_qs());

    cout << "Testing unpacker integrity" << endl;
    flexmat V1;
    V1.init(iSetup.vValsVhhhh, iSetup.iVhhhh, iSetup.jVhhhh, iSetup.kVhhhh, iSetup.lVhhhh, iSetup.iNh, iSetup.iNh, iSetup.iNh, iSetup.iNh);
    V1.update(vhhhh.pq_rs());


    for(int i = 0; i<iSetup.iNh; i++){
        for(int j = 0; j<iSetup.iNh; j++){
            for(int k = 0; k<iSetup.iNh; k++){
                for(int l = 0; l<iSetup.iNh; l++){
                    if(V1.pq_rs()(i + j*iSetup.iNh, k+l*iSetup.iNh) != vhhhh.pq_rs()(i + j*iSetup.iNh, k+l*iSetup.iNh)){
                        cout << "Inconsistencies found:" << i << " " << j << " " << k << " " << l << "     " << V1.pq_rs()(i + j*iSetup.iNh, k+l*iSetup.iNh) << "     " << vhhhh.pq_rs()(i + j*iSetup.iNh, k+l*iSetup.iNh) << endl;
                    }
                }
            }
        }
    }

    /*
    for(int i = 0; i < V1.vValues.size(); i++){
        cout << V1.pq_rs().row_indices[i] << "        " << vhhhh.pq_rs().row_indices[i] << endl;
    }

    for(int i = 0; i < V1.pq_rs().n_cols; i++){
        cout << V1.pq_rs().col_ptrs[i] << "        " << vhhhh.pq_rs().col_ptrs[i] << endl;
    }
    */


    //cout << V1.pq_rs().n_nonzero << endl;
    //cout << vhhhh.pq_rs().n_nonzero << endl;
    cout << V1.vValues.size() << " " << vhhhh.vValues.size() << endl;




    //Vhh.pq_rs().print();
    //Vhh.p_qrs().print();

    /*
    //Vhh.init(iSetup.vValsVhhhh, iSetup.iVhhhh, iSetup.jVhhhh, iSetup.kVhhhh, iSetup.lVhhhh, iSetup.iNh, iSetup.iNh, iSetup.iNh, iSetup.iNh);
    iSetup.sVhhhh();
    flexmat Vhhhh;
    Vhhhh.init(iSetup.vValsVhhhh, iSetup.iVhhhh, iSetup.jVhhhh, iSetup.kVhhhh, iSetup.lVhhhh, iSetup.iNh, iSetup.iNh, iSetup.iNh, iSetup.iNh);

    iSetup.sVpppp();
    flexmat Vpppp;
    Vpppp.init(iSetup.vValsVpppp, iSetup.aVpppp, iSetup.bVpppp, iSetup.cVpppp, iSetup.dVpppp, iSetup.iNp, iSetup.iNp, iSetup.iNp, iSetup.iNp);
    //Vpppp.pq_rs().print();

    iSetup.sVhpph();
    flexmat Vhpph;
    Vhpph.init(iSetup.vValsVhpph, iSetup.iVhpph, iSetup.aVhpph, iSetup.bVhpph, iSetup.jVhpph, iSetup.iNh, iSetup.iNp, iSetup.iNp, iSetup.iNh);

    iSetup.sVhhpp();
    flexmat Vhhpp;
    Vhhpp.init(iSetup.vValsVhhpp, iSetup.iVhhpp, iSetup.jVhhpp, iSetup.aVhhpp, iSetup.bVhhpp, iSetup.iNh, iSetup.iNh, iSetup.iNp, iSetup.iNp);

    //initialize amplitude
    flexmat T;
    T.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    T.set_amplitudes(bs.vEnergy);
    //T.pq_rs().print();

    sp_mat cv = Vhhpp.pq_rs() * T.pq_rs();
    mat Cv(cv);
    double C_ = 0;
    for(int i = 0; i<Cv.n_cols; i++){
        C_+= Cv(i,i);
    }
    cout << .25*C_ << endl;
    energy();

    */
    energy();
}

void ccd::energy(){
    //Calculate the ground state energy
    sp_mat cv = vhhpp.pq_rs() * T.pq_rs();
    mat Cv(cv);
    double C_ = 0;
    for(int i = 0; i<Cv.n_cols; i++){
        C_+= Cv(i,i);
    }
    cout << .25*C_ << endl;

}
