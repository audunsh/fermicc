#include "ccd.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "solver/flexmat.h"
#include "basis/electrongas.h"
#include "solver/initializer.h"
#include "solver/unpack_sp_mat.h"
#include <time.h>

#include <eigen/Eigen/Dense>
#include <eigen/Eigen/Sparse>

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
    vhhhh.shed_zeros();
    vpppp.init(iSetup.vValsVpppp, iSetup.aVpppp, iSetup.bVpppp, iSetup.cVpppp, iSetup.dVpppp, iSetup.iNp, iSetup.iNp, iSetup.iNp, iSetup.iNp);
    vpppp.shed_zeros();
    vhpph.init(iSetup.vValsVhpph, iSetup.iVhpph, iSetup.aVhpph, iSetup.bVhpph, iSetup.jVhpph, iSetup.iNh, iSetup.iNp, iSetup.iNp, iSetup.iNh);
    vhpph.shed_zeros();
    vhhpp.init(iSetup.vValsVhhpp, iSetup.iVhhpp, iSetup.jVhhpp, iSetup.aVhhpp, iSetup.bVhhpp, iSetup.iNh, iSetup.iNh, iSetup.iNp, iSetup.iNp);
    vhhpp.shed_zeros();
    vpphh.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    vpphh.shed_zeros();

    //set up first T2-amplitudes
    T.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    T.set_amplitudes(bs.vEnergy);


    // HOW TO SET UP FLEXMAT OBJECTS FROM CSC-MATRICES
    // flexmat V1;
    // V1.update(vhhhh.pq_rs(),vhhhh.iNp, vhhhh.iNq, vhhhh.iNr, vhhhh.iNs); //update (or initialize) with an sp_mat object (requires unpacking)

    //clock_t t;
    //sp_mat H;
    //H = vpppp.pq_rs();
    //t = clock();
    //H = H*H;
    //cout << "Spent " << ((float)t)/CLOCKS_PER_SEC << " seconds on multiplication."<< endl;

    //testing the sparselibrary in Eigen
    typedef Eigen::Triplet<double> Tr;
    typedef Eigen::SparseMatrix<double> SparseMat;

    clock_t t1, t0, t2;
    t0 = clock();

    std::vector<Tr> tripletList;
    tripletList.reserve(vpppp.vValues.size());
    for(int i= 0; i<vpppp.vValues.size(); i++){
        tripletList.push_back(Tr(vpppp.vp(i)+vpppp.vq(i)*vpppp.iNp, vpppp.vr(i)+vpppp.vs(i)*vpppp.iNr, vpppp.vValues(i)));
    }
    SparseMat Vp1(vpppp.iNp*vpppp.iNq, vpppp.iNr*vpppp.iNs);
    Vp1.setFromTriplets(tripletList.begin(), tripletList.end());

    std::vector<Tr> tripletList2;
    tripletList2.reserve(vpphh.vValues.size());
    for(int i= 0; i<vpphh.vValues.size(); i++){
        tripletList2.push_back(Tr(vpphh.vp(i)+vpphh.vq(i)*vpphh.iNp, vpphh.vr(i)+vpphh.vs(i)*vpphh.iNr, vpphh.vValues(i)));
    }
    SparseMat Tp1(vpphh.iNp*vpphh.iNq, vpphh.iNr*vpphh.iNs);
    Tp1.setFromTriplets(tripletList2.begin(), tripletList2.end());
    t1 = clock();


    SparseMat Sp3 = Vp1*Tp1;
    t2 = clock();
    cout << "Eigen (Setup)         :" << t1-t0 << endl;
    cout << "Eigen (multiplication):" << t2-t1 << endl;
    cout << "Eigen (total)         :" << t2-t0 << endl;

    t0 = clock();
    sp_mat vp12 = vpppp.pq_rs();
    vp12 = vpphh.pq_rs();

    t1 = clock();
    vp12 = vpppp.pq_rs()*vpphh.pq_rs();
    t2 = clock();

    cout << "Armadillo (Setup)         :" << t1-t0 << endl;
    cout << "Armadillo (multiplication):" << t2-t1 << endl;
    cout << "Armadillo (total)         :" << t2-t0 << endl;





    cout << CLOCKS_PER_SEC << endl;

    energy();
    //for(int i = 0; i < 20; i++){
    //    advance();
    //}
    //energy();
}

void ccd::advance(){
    //advance the solution one step
    int Np = iSetup.iNp;
    int Nq = iSetup.iNp;
    int Nr = iSetup.iNh;
    int Ns = iSetup.iNh;

    L1 = vpppp.pq_rs()*T.pq_rs();
    L2 = T.pq_rs()*vhhhh.pq_rs();

    fmL3.update(vhpph.sq_rp()*T.qs_pr(), Ns, Nq, Np, Nr);
    L3 = fmL3.rq_sp() - fmL3.qr_sp() - fmL3.rq_ps() + fmL3.qr_ps();
    //L3 *= 0;

    //L3 = fmL3.sq_pr()-fmL3.sp_qr()-fmL3.rq_ps()+fmL3.rp_qs(); //permuting elements

    Q1 = T.pq_rs()*vhhpp.pq_rs()*T.pq_rs();

    fmQ2.update(T.pr_qs()*vhhpp.rp_qs()*T.sq_pr(), Np, Nr, Nq, Ns); //needs realignment and permutations
    Q2 = fmQ2.pr_qs()-fmQ2.pr_sq(); //permuting elements

    fmQ3.update_as_pqs_r(T.pqs_r()*vhhpp.q_prs()*T.sqp_r(), Np, Nq, Nr, Ns); //needs realignment and permutations
    Q3 = fmQ3.pq_rs() - fmQ3.pq_sr(); //permuting elements

    fmQ4.update_as_p_qrs(T.p_srq()*vhhpp.pqr_s()*T.p_qrs(), Np, Nq, Nr, Ns); //needs realignment and permutations
    Q4 = fmQ4.pq_rs() - fmQ4.qp_rs(); //permuting elements

    T.update(vpphh.pq_rs() + .5*(L1 + L2) + L3 + .25*Q1 + Q2 - .5*Q3 - .5*Q4, Np, Nq, Nr, Ns);
    T.set_amplitudes(ebs.vEnergy); //divide updated amplitides by energy denominator
    energy();
}

void ccd::energy(){
    //Calculate the ground state energy
    sp_mat cv = vhhpp.pq_rs() * T.pq_rs();
    mat Cv(cv); //this is inefficient: does not utilize sp_mat functionality, one possibility through unpack_sp_mat
    double C_ = 0;
    for(int i = 0; i<Cv.n_cols; i++){
        C_+= Cv(i,i);
    }
    cout << "E1:" << .25*C_ << endl;

    double dC = accu(vhhpp.pq_rs()*T.pq_rs());
    cout << "E2:" << .25*dC << endl;

}
