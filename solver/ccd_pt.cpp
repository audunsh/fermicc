#include "ccd_pt.h"

#define ARMA_64BIT_WORD
#include <armadillo>
#include "solver/flexmat.h"
#include "solver/flexmat6.h"

#include "basis/electrongas.h"
#include "solver/initializer.h"
#include "solver/unpack_sp_mat.h"
#include <time.h>

//#include <eigen/Eigen/Dense>
//#include <eigen/Eigen/Sparse>

using namespace std;
using namespace arma;

ccd_pt::ccd_pt(electrongas bs){
    iterations = 0; //current number of iterations
    ebs = bs;
    iSetup = initializer(bs);



    //setup all interaction matrices
    iSetup.sVhhhhO();
    iSetup.sVppppBlock();
    iSetup.sVhhpp();
    iSetup.sVhpph();



    //convert interaction data to flexmat objects
    vhhhh.init(iSetup.vValsVhhhh, iSetup.iVhhhh, iSetup.jVhhhh, iSetup.kVhhhh, iSetup.lVhhhh, iSetup.iNh, iSetup.iNh, iSetup.iNh, iSetup.iNh);
    vhhhh.shed_zeros();
    vhpph.init(iSetup.vValsVhpph, iSetup.iVhpph, iSetup.aVhpph, iSetup.bVhpph, iSetup.jVhpph, iSetup.iNh, iSetup.iNp, iSetup.iNp, iSetup.iNh);
    vhpph.shed_zeros();
    vhhpp.init(iSetup.vValsVhhpp, iSetup.iVhhpp, iSetup.jVhhpp, iSetup.aVhhpp, iSetup.bVhhpp, iSetup.iNh, iSetup.iNh, iSetup.iNp, iSetup.iNp);
    vhhpp.shed_zeros();
    vpphh.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    vpphh.shed_zeros();


    //triples specific
    iSetup.sVhppp();
    iSetup.sVhphh();

    vhppp.init(iSetup.vValsVhppp, iSetup.iVhppp, iSetup.aVhppp, iSetup.bVhppp, iSetup.cVhppp, iSetup.iNh, iSetup.iNp, iSetup.iNp, iSetup.iNp);
    vphpp.init(-iSetup.vValsVhppp, iSetup.aVhppp, iSetup.iVhppp, iSetup.bVhppp, iSetup.cVhppp, iSetup.iNp, iSetup.iNh, iSetup.iNp, iSetup.iNp);

    vhphh.init(iSetup.vValsVhphh, iSetup.iVhphh, iSetup.aVhphh, iSetup.jVhphh, iSetup.kVhphh, iSetup.iNh, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    vhhhp.init(iSetup.vValsVhhhp, iSetup.iVhhhp, iSetup.jVhhhp, iSetup.kVhhhp, iSetup.aVhhhp, iSetup.iNh, iSetup.iNh, iSetup.iNh, iSetup.iNp);

    vhppp.shed_zeros();
    vphpp.shed_zeros();
    vhphh.shed_zeros();

    vppph.shed_zeros();
    vhhhp.shed_zeros();
    //vhhhp.pqs_r()*2;

    //cout << vhhhp.vValues.max() << endl;

    vhppp.report();
    vphpp.report();

    vhphh.report();
    vhhhp.report();



    //set up first T2-amplitudes
    T.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    T.shed_zeros();
    T.set_amplitudes(bs.vEnergy);
    T.map_indices();

    //check_matrix_consistency();
    energy();
    for(int i = 0; i < 25; i++){
        iterations += 1;
        advance();
    }

}

void ccd_pt::check_matrix_consistency(){
    //Check that all elements in matrices correspond to the interaction given in the basis
    //NOTE: This is a time consuming process, especially for large basis sets

    int vpphh_err = 0;
    int vhhpp_err = 0;
    int tpphh_err = 0;
    for(int a = 0; a < iSetup.iNp; a++){
        for(int b = 0; b < iSetup.iNp; b++){
            for(int i = 0; i < iSetup.iNh; i++){
                for(int j = 0; j < iSetup.iNh; j++){

                    if(vpphh.pq_rs()(a + b*iSetup.iNp, i + j*iSetup.iNh) != iSetup.bs.v2(a + iSetup.iNh , b+ iSetup.iNh, i,j)){
                        //cout << "Found discrepancy" << endl;
                        vpphh_err += 1;
                    }
                    if(vhhpp.pq_rs()(i+j*iSetup.iNh, a + b*iSetup.iNp) != iSetup.bs.v2(i,j,a + iSetup.iNh , b+ iSetup.iNh)){
                        //cout << "Found discrepancy" << endl;
                        vhhpp_err += 1;
                    }
                    if(T.pq_rs()(a + b*iSetup.iNp, i + j*iSetup.iNh) != iSetup.bs.v2(a + iSetup.iNh , b+ iSetup.iNh, i,j)/(iSetup.bs.vEnergy(i) + iSetup.bs.vEnergy(j)-iSetup.bs.vEnergy(a+iSetup.iNh)-iSetup.bs.vEnergy(b+iSetup.iNh))){
                        cout << "Found discrepancy" << T.pq_rs()(a + b*iSetup.iNp, i + j*iSetup.iNh)<< iSetup.bs.v2(a + iSetup.iNh , b+ iSetup.iNh, i,j)/(iSetup.bs.vEnergy(i) + iSetup.bs.vEnergy(j)-iSetup.bs.vEnergy(a)-iSetup.bs.vEnergy(b))<< endl;
                        tpphh_err += 1;
                    }
                }
            }
        }
    }
    cout << "Found " << vpphh_err << " inconsistent elements in vpphh." << endl;
    cout << "Found " << vhhpp_err << " inconsistent elements in vhhpp." << endl;
    cout << "Found " << tpphh_err << " inconsistent elements in thhpp." << endl;
}

void ccd_pt::L1_dense_multiplication(){
    L1.clear();

    uint N = iSetup.bmVpppp.uN; //number of blocks
    int Nh = iSetup.iNh;

    field<uvec> stream;

    vec vals;
    umat coo;
    uint a,b,c,d, Na;
    uint total_elements = 0; //total number of elements calculated;

    mat tempStorage, Ttemp;
    double val;

    field<umat> fmLocations(N);
    field<vec> fvValues(N);

    mat V;

    for(uint i = 0; i < N; ++i){
        V.clear();
        stream = iSetup.bmVpppp.get_block(i); //get current block
        Na = stream(0).size();
        V.set_size(Na, Na);
        for(uint p = 0; p<Na; ++p){
            a = stream(0)(p);
            b = stream(1)(p);
            //NOTE: Actually going through the kroenecker deltas here, possible to skip many tests in the interaction p==r, q==s
            //Interaction below has already passed d(k_p+k_q, k_r + k_s) && m_p==m_r && m_q == m_s
            //val = iSetup.bs.v2(a+Nh,b+Nh);
            val = iSetup.bs.v2(a+Nh,b+Nh,a+Nh,b+Nh);


            V(p,p) = val;


            //Interaction below has already passed d(k_p+k_q, k_r + k_s)
            for(uint q = p+1; q<Na; ++q){
                c = stream(2)(q);
                d = stream(3)(q);

                val = iSetup.bs.v2(a+Nh,b+Nh,c+Nh,d+Nh); //create separate function here

                V(p,q) = val;
                V(q,p) = val;
            }
        }

        //perform multiplication and cast to sparse matrix L1;
        Ttemp = T.rows_dense(stream(4)); //load only elements in row
        tempStorage = V*Ttemp;

        int N_elems = T.MCols.size()*stream(4).size();
        umat locations(2, N_elems);
        vec values(N_elems);



        uint count = 0;
        for(uint p = 0; p<stream(4).size(); ++p ){
            for(uint q = 0; q<T.MCols.size(); ++q ){
                locations(0, count) = stream(4)(p);
                locations(1, count) = T.MCols(q);
                values(count) = tempStorage(p,q);
                count += 1;
            }
        }


        fmLocations(i) = locations;
        fvValues(i) = values;
        total_elements += count;
    }
    //create sparse L1 from each block
    umat mCOO(2, total_elements);
    vec vData(total_elements);
    int iCount=0;
    for(uint i = 0; i < N; ++i){
        for(uint j = 0; j < fvValues(i).size(); ++j){
            mCOO(0, iCount) = fmLocations(i)(0,j);
            mCOO(1, iCount) = fmLocations(i)(1,j);
            vData(iCount) = fvValues(i)(j);
            iCount += 1;
        }
    }
    L1 = sp_mat(mCOO, vData, iSetup.iNp*iSetup.iNp,iSetup.iNh*iSetup.iNh);
}


void ccd_pt::advance(){
    //advance the solution one step
    int Np = iSetup.iNp;
    int Nq = iSetup.iNp;
    int Nr = iSetup.iNh;
    int Ns = iSetup.iNh;
    bool timing = false; //time each contribution calculation and print to screen (each iteration)

    L1_dense_multiplication();

    L2 = T.pq_rs()*vhhhh.pq_rs();

    fmL3.update(vhpph.sq_rp()*T.qs_pr(), Ns, Nq, Np, Nr);
    L3 = fmL3.rq_sp() - fmL3.qr_sp() -fmL3.rq_ps() +fmL3.qr_ps();

    fmQ1.update(T.rs_pq()*vhhpp.rs_pq()*T.rs_pq(), Nr, Ns, Np,Nq);
    Q1 = fmQ1.rs_pq();

    fmQ2.update(T.pr_qs()*vhhpp.rp_qs()*T.sq_pr(), Np, Nr, Nq, Ns); //needs realignment and permutations
    Q2 = fmQ2.pr_qs()-fmQ2.pr_sq(); //permuting elements

    fmQ3.update_as_r_pqs((T.r_sqp()*vhhpp.prs_q())*T.r_pqs(), Np, Nq, Nr, Ns); //needs realignment and permutations
    Q3 = fmQ3.pq_rs() - fmQ3.pq_sr(); //permuting elements

    fmQ4.update_as_p_qrs(T.p_srq()*vhhpp.pqr_s()*T.p_qrs(), Np, Nq, Nr, Ns); //needs realignment and permutations
    Q4 = fmQ4.pq_rs() - fmQ4.qp_rs(); //permuting elements

    //Calculating perturbative triples amplitudes

    t2a.update_as_qru_pst(vppph.pqs_r()*T.q_prs(), Np,Np,Np,Nr,Nr,Nr);
    t2a.update_as_pqr_stu(t2a.pqr_stu()-t2a.qpr_stu()-t2a.rpq_stu()-t2a.rpq_uts()+t2a.prq_stu()+t2a.qrp_uts()-t2a.qrp_ust()+t2a.rqp_uts()+t2a.pqr_ust(), Np,Np,Np,Nr,Nr,Nr);

    t2b.update_as_pqs_rtu(T.pqr_s()*vhphh.p_qrs(), Np,Np,Np,Nr,Nr,Nr);
    t2b.update_as_pqr_stu(t2b.pqr_stu()-t2b.rqp_stu()-t2b.rpq_stu()-t2b.rpq_tsu()+t2b.qpr_stu()+t2b.qrp_tsu()-t2b.qrp_ust()+t2b.prq_tsu()+t2b.pqr_ust(), Np,Np,Np,Nr,Nr,Nr);
    //T3.deinit();
    T3.update_as_pqr_stu(t2a.pqr_stu() - t2b.pqr_stu(), Np,Np,Np,Nr,Nr,Nr);

    //Calculating the triples contributions to T2
    fmD10b.update_as_q_rsp(vphpp.p_qrs()*T3.uqr_stp(), Np,Np,Nr,Nr);
    fmD10b.update(fmD10b.pq_rs() - fmD10b.qp_rs(), Np, Nq, Nr, Ns);

    fmD10c.update_as_pqr_s(T3.pqs_tur()*vhhhp.pqs_r(), Np,Np,Nr,Nr); //remember to permute these


    //fmD10c.update_as_pqr_s(T3.pqs_tur()*vhphh.srp_q(), Np,Np,Nr,Nr); //remember to permute these
    fmD10c.update(fmD10c.pq_rs() - fmD10c.pq_sr(), Np,Np,Nr,Nr);

    T.update(vpphh.pq_rs() + .5*(L1 + L2) + L3 + .25*Q1 + Q2 - .5*Q3 - .5*Q4 + .5*(fmD10b.pq_rs() - fmD10c.pq_rs()), Np, Nq, Nr, Ns);
    //T.update(vpphh.pq_rs() + .5*(L1 + L2) + L3 + .25*Q1 + Q2 - .5*Q3 - .5*Q4 , Np, Nq, Nr, Ns);


    T.set_amplitudes(ebs.vEnergy); //divide updated amplitides by energy denominator

    energy();
    T.shed_zeros();
    T.map_indices();
}




void ccd_pt::energy(){
    //Calculate the ground state energy
    sp_mat cv = vhhpp.pq_rs() * T.pq_rs();
    //mat vhhpp2(vhhpp.pq_rs());
    //mat tpphh2(T.pq_rs());
    //mat Cv = vhhpp2*tpphh2;
    mat Cv(cv); //this is inefficient: does not utilize sp_mat functionality, one possibility through unpack_sp_mat
    double C_ = 0;
    for(int i = 0; i<Cv.n_cols; i++){
        C_+= Cv(i,i);
    }


    correlation_energy = .25*C_;
    cout << "[CCD_pt]["  << iterations  << "]" << "Energy               :" << .25*C_ << endl;
    cout << "[CCD_pt]["  << iterations  << "]" << "Energy (per particle):" << .25*C_/iSetup.iNh << endl;

}
