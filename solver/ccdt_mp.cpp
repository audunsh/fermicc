#include "ccdt_mp.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "solver/flexmat.h"
#include "solver/flexmat6.h"

#include "basis/electrongas.h"
#include "solver/initializer.h"
#include "solver/unpack_sp_mat.h"
#include <omp.h>
#include <time.h>

//#include <eigen/Eigen/Dense>
//#include <eigen/Eigen/Sparse>

using namespace std;
using namespace arma;

ccdt_mp::ccdt_mp(electrongas bs, double a){
    // ##################################################
    // ##                                              ##
    // ## Full CCDT, initialization                    ##
    // ##                                              ##
    // ##################################################

    alpha = a;  //relaxation parameter
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
    vhhhp.shed_zeros();

    //set up first T2-amplitudes
    T.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    T.shed_zeros();
    T.set_amplitudes(bs.vEnergy);
    T.map_indices();


    sp_mat Y;
    T3.update_as_pqr_stu(Y, iSetup.iNp, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh, iSetup.iNh);
    T3.shed_zeros();
    T3.map_indices();
    //check_matrix_consistency();
    energy();
    for(int i = 0; i < 25; i++){
        iterations += 1;
        advance();
    }

}

void ccdt_mp::check_matrix_consistency(){
    // ##################################################
    // ##                                              ##
    // ## Matrix consistency test, debugging function  ##
    // ##                                              ##
    // ##################################################

    //This function checks that all elements in matrices correspond to the interaction given in the basis
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
                        //cout << "Found discrepancy" << T.pq_rs()(a + b*iSetup.iNp, i + j*iSetup.iNh)<< iSetup.bs.v2(a + iSetup.iNh , b+ iSetup.iNh, i,j)/(iSetup.bs.vEnergy(i) + iSetup.bs.vEnergy(j)-iSetup.bs.vEnergy(a)-iSetup.bs.vEnergy(b))<< endl;
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

void ccdt_mp::L1_dense_multiplication(){
    // #######################################################
    // ##                                                   ##
    // ##  Calculate diagrams containing pppp interactions  ##
    // ##  Limit memory usage, calculate terms on the fly   ##
    // ##  Further optimization of this routine is possible ##
    // ##                                                   ##
    // #######################################################

    L1.clear();

    uint N = iSetup.bmVpppp.uN; //number of blocks
    int Nh = iSetup.iNh;

    field<uvec> stream;

    vec vals;
    umat coo;
    uint a,b,c,d, Na;
    uint total_elements_L1 = 0; //total number of elements calculated;
    uint total_elements_t3a = 0; //total number of elements calculated;


    mat L1_tempStorage, L1_Ttemp;
    mat t3a_tempStorage, t3a_Ttemp;

    double val;

    field<umat> fmLocations_L1(N);
    field<vec> fvValues_L1(N);

    field<umat> fmLocations_t3a(N);
    field<vec> fvValues_t3a(N);

    mat V;

    for(uint i = 0; i < N; ++i){
        // #############################################
        // ##   Iteration over each block in vhpppp   ##
        // #############################################
        V.clear();
        stream = iSetup.bmVpppp.get_block(i); //get indices in current block
        Na = stream(0).size();
        V.set_size(Na, Na);
        for(uint p = 0; p<Na; ++p){
            a = stream(0)(p);
            b = stream(1)(p);
            //NOTE: Actually going through the kroenecker deltas here, possible to skip many tests in the interaction p==r, q==s
            //Interaction below has already passed d(k_p+k_q, k_r + k_s) && m_p==m_r && m_q == m_s
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
        // ################################################
        // ##   Fetch corresponding rows in amplitudes   ##
        // ##   Perform dense multiplication    (L1)     ##
        // ################################################
        L1_Ttemp = T.rows_dense(stream(4)); //load only elements in row, keep
        L1_tempStorage = V*L1_Ttemp; //dense multiplication
        int N_elems = T.MCols.size()*stream(4).size();
        umat locations(2, N_elems);
        vec values(N_elems);
        uint count = 0;
        for(uint p = 0; p<stream(4).size(); ++p ){
            for(uint q = 0; q<T.MCols.size(); ++q ){
                locations(0, count) = stream(4)(p);
                locations(1, count) = T.MCols(q);
                values(count) = L1_tempStorage(p,q);
                count += 1;
            }
        }
        // #################################
        // ##   Store partial results     ##
        // #################################
        fmLocations_L1(i) = locations;
        fvValues_L1(i) = values;
        total_elements_L1 += count;

        // ################################################
        // ##   Fetch corresponding rows in amplitudes   ##
        // ##   Perform dense multiplication    (t3a)    ##
        // ################################################
        t3a_Ttemp = T3.rows_dense(stream(4)); //load only elements in row, keep
        t3a_tempStorage = V*t3a_Ttemp; //dense multiplication
        N_elems = T3.MCols.size()*stream(4).size();
        locations.set_size(2, N_elems);
        values.set_size(N_elems);
        count = 0;
        for(uint p = 0; p<stream(4).size(); ++p ){
            for(uint q = 0; q<T3.MCols.size(); ++q ){
                locations(0, count) = stream(4)(p);
                locations(1, count) = T3.MCols(q);
                values(count) = t3a_tempStorage(p,q);
                count += 1;
            }
        }
        // #################################
        // ##   Store partial results(t3a)##
        // #################################
        fmLocations_t3a(i) = locations;
        fvValues_t3a(i) = values;
        total_elements_t3a += count;
    }
    // #########################
    // ## construct diagram L1##
    // #########################

    umat mCOO(2, total_elements_L1);
    vec vData(total_elements_L1);
    int iCount=0;
    for(uint i = 0; i < N; ++i){
        for(uint j = 0; j < fvValues_L1(i).size(); ++j){
            mCOO(0, iCount) = fmLocations_L1(i)(0,j);
            mCOO(1, iCount) = fmLocations_L1(i)(1,j);
            vData(iCount) = fvValues_L1(i)(j);
            iCount += 1;
        }
    }
    L1 = sp_mat(mCOO, vData, iSetup.iNp*iSetup.iNp,iSetup.iNh*iSetup.iNh);

    // ###########################
    // ## construct diagram t3a ##
    // ###########################

    mCOO.set_size(2, total_elements_t3a);
    vData.set_size(total_elements_t3a);
    iCount=0;
    for(uint i = 0; i < N; ++i){
        for(uint j = 0; j < fvValues_t3a(i).size(); ++j){
            mCOO(0, iCount) = fmLocations_t3a(i)(0,j);
            mCOO(1, iCount) = fmLocations_t3a(i)(1,j);
            vData(iCount) = fvValues_t3a(i)(j);
            iCount += 1;
        }
    }
    t3a.update_as_pqr_stu(sp_mat(mCOO, vData, iSetup.iNp*iSetup.iNp*iSetup.iNp,iSetup.iNh*iSetup.iNh*iSetup.iNh), iSetup.iNp,iSetup.iNp,iSetup.iNp,iSetup.iNh,iSetup.iNh,iSetup.iNh);
}


void ccdt_mp::advance(){
    //advance the solution one step
    int Np = iSetup.iNp;
    int Nq = iSetup.iNp;
    int Nr = iSetup.iNh;
    int Ns = iSetup.iNh;
    bool timing = true; //time each contribution calculation and print to screen (each iteration)

    double tm = omp_get_wtime();

    // ##################################################
    // ##                                              ##
    // ## Calculating doubles amplitude                ##
    // ##                                              ##
    // ##################################################

    L1_dense_multiplication(); //The pp-pp diagrama, given special treatment to limit memory usage

    L2 = T.pq_rs()*vhhhh.pq_rs();

    fmL3.update(vhpph.sq_rp()*T.qs_pr(), Ns, Nq, Np, Nr);
    L3 = fmL3.rq_sp() - fmL3.qr_sp() -fmL3.rq_ps() +fmL3.qr_ps(); //permuting elements

    fmQ1.update(T.rs_pq()*vhhpp.rs_pq()*T.rs_pq(), Nr, Ns, Np,Nq);
    Q1 = fmQ1.rs_pq();

    fmQ2.update(T.pr_qs()*vhhpp.rp_qs()*T.sq_pr(), Np, Nr, Nq, Ns);
    Q2 = fmQ2.pr_qs()-fmQ2.pr_sq(); //permuting elements

    fmQ3.update_as_r_pqs((T.r_sqp()*vhhpp.prs_q())*T.r_pqs(), Np, Nq, Nr, Ns);
    Q3 = fmQ3.pq_rs() - fmQ3.pq_sr(); //permuting elements

    fmQ4.update_as_p_qrs(T.p_srq()*vhhpp.pqr_s()*T.p_qrs(), Np, Nq, Nr, Ns);
    Q4 = fmQ4.pq_rs() - fmQ4.qp_rs(); //permuting elements

    if(timing){
        cout << "Doubles + Ladders:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }

    // ##################################################
    // ##                                              ##
    // ## Calculating full triples amplitudes          ##
    // ##                                              ##
    // ##################################################

    // To do: restructure the progression of this part, so that a minimum of terms needs to be stored.

    // #########################
    // ##   Linear t2 terms   ##
    // #########################

    t2a.update_as_qru_pst(vppph.pqs_r()*T.q_prs(), Np,Np,Np,Nr,Nr,Nr);
    t2a.update_as_pqr_stu(t2a.pqr_stu()-t2a.qpr_stu()-t2a.rpq_stu()-t2a.rpq_uts()+t2a.prq_stu()+t2a.qrp_uts()-t2a.qrp_ust()+t2a.rqp_uts()+t2a.pqr_ust(), Np,Np,Np,Nr,Nr,Nr);

    t2b.update_as_pqs_rtu(T.pqr_s()*vhphh.p_qrs(), Np,Np,Np,Nr,Nr,Nr);
    t2b.update_as_pqr_stu(t2b.pqr_stu()-t2b.rqp_stu()-t2b.rpq_stu()-t2b.rpq_tsu()+t2b.qpr_stu()+t2b.qrp_tsu()-t2b.qrp_ust()+t2b.prq_tsu()+t2b.pqr_ust(), Np,Np,Np,Nr,Nr,Nr);

    if(timing){
        cout << "Linear t2 terms:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }

    // #########################
    // ##  Linear t3 terms    ##
    // #########################

    t3b.update_as_pqru_st(T3.pqrs_tu()*vhhhh.pq_rs(), Np,Np,Np,Nr,Nr,Nr); //replaced interaction (symmetries)
    t3b.update_as_pqr_stu(t3b.pqr_stu()-t3b.pqr_sut()-t3b.pqr_tus(), Np, Np, Np, Nr, Nr, Nr);

    t3c.update_as_ps_qrtu(vhpph.qs_pr()*T3.sp_qrtu(), Np,Np,Np,Nr,Nr,Nr);
    t3c.update_as_pqr_stu(t3c.pqr_stu()-t3c.qpr_stu()-t3c.rpq_stu()-t3c.rpq_tsu()+t3c.prq_stu()+t3c.qrp_tsu()-t3c.qrp_ust()+t3c.rqp_tsu()+t3c.pqr_ust(), Np, Np, Np, Nr, Nr, Nr);

    if(timing){
        cout << "Linear t3 terms:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }

    // #########################
    // ## Mixed t2*t3 terms   ##
    // #########################

    if(timing){
        cout << "Linear t3 terms:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }
    t2t3a.update_as_qtru_ps(T3.qtru_sp()*vhhpp.qs_pr()*T.sq_pr(), Np, Np, Np, Nr, Nr, Nr );
    t2t3a.update_as_pqr_stu(t2t3a.pqr_stu()-t2t3a.qpr_stu()-t2t3a.rpq_stu()-t2t3a.rpq_tsu()+t2t3a.prq_stu()+t2t3a.qrp_tsu()-t2t3a.qrp_ust()+t2t3a.rqp_tsu()+t2t3a.pqr_ust(), Np, Np, Np, Nr, Nr, Nr);

    if(timing){
        cout << "Mixed t2t3a:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }
    t2t3b.update_as_pqtru_s(T3.pqtru_s()*(vhhpp.q_prs()*T.rpq_s()), Np, Np, Np, Nr, Nr, Nr);
    t2t3b.update_as_pqr_stu(t2t3b.pqr_stu()-t2t3b.pqr_tsu()-t2t3b.pqr_ust(), Np, Np, Np, Nr, Nr, Nr);

    if(timing){
        cout << "Mixed t2t3b:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }
    t2t3c.update_as_sqtru_p(T3.sqtru_p()*(vhhpp.s_pqr()*T.rsp_q()), Np, Np, Np, Nr, Nr, Nr);
    t2t3c.update_as_pqr_stu(t2t3c.pqr_stu()-t2t3c.qpr_stu()-t2t3c.rpq_stu(), Np, Np, Np, Nr, Nr, Nr);

    if(timing){
        cout << "Mixed t2t3c:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }
    t2t3d.update_as_qru_pst(T3.pru_stq()*vhhpp.pqs_r()*T.q_prs(), Np, Np, Np, Nr, Nr, Nr);
    t2t3d.update_as_pqr_stu(t2t3d.pqr_stu()-t2t3d.qpr_stu()-t2t3d.rpq_stu()-t2t3d.rpq_sut()+t2t3d.prq_stu()+t2t3d.qrp_sut()-t2t3d.qrp_tus()+t2t3d.rqp_sut()+t2t3d.pqr_tus(), Np, Np, Np, Nr, Nr, Nr);

    if(timing){
        cout << "Mixed t2t3d:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }
    t2t3e.update_as_tru_pqs((T3.sru_tpq()*vhhpp.qrs_p())*T.s_pqr(), Np, Np, Np, Nr, Nr, Nr);
    t2t3e.update_as_pqr_stu(t2t3e.pqr_stu()-t2t3e.rqp_stu()-t2t3e.rpq_stu()-t2t3e.rpq_tsu()+t2t3e.qpr_stu()+t2t3e.qrp_tsu()-t2t3e.qrp_ust()+t2t3e.prq_tsu()+t2t3e.pqr_ust(), Np, Np, Np, Nr, Nr, Nr);

    if(timing){
        cout << "Mixed t2t3e:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }
    t2t3f.update_as_pqru_st(T3.pqru_st()*vhhpp.pq_rs()*T.pq_rs(), Np, Np, Np, Nr, Nr, Nr);
    t2t3f.update_as_pqr_stu(t2t3f.pqr_stu()-t2t3f.pqr_uts()-t2t3f.pqr_ust(), Np, Np, Np, Nr, Nr, Nr);

    if(timing){
        cout << "Mixed t2t3f:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }
    t2t3g.update_as_stru_pq(T3.stru_pq()*vhhpp.rs_pq()*T.rs_pq(), Np, Np, Np, Nr, Nr, Nr);
    t2t3g.update_as_pqr_stu(t2t3g.pqr_stu()-t2t3g.rqp_stu()-t2t3g.rpq_stu(), Np, Np, Np, Nr, Nr, Nr);

    if(timing){
        cout << "Mixed t2t3g:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }

    // #########################
    // ##  Quadratic t2 terms ##
    // #########################

    // These terms need special treatment due to lines exciting out of the interaction

    fmt2temp.update(T.pr_sq()*vhppp.pr_qs(), Np, Nr, Np, Np);
    t2t2b.update_as_psq_rtu(fmt2temp.pqr_s()*T.p_qrs(), Np,Np,Np,Nr,Nr,Nr);
    t2t2b.update_as_pqr_stu(t2t2b.pqr_stu()-t2t2b.qpr_stu()-t2t2b.rpq_stu()-t2t2b.rpq_tsu()+t2t2b.prq_stu()+t2t2b.qrp_tsu()-t2t2b.qrp_ust()+t2t2b.rqp_tsu()+t2t2b.pqr_ust(), Np,Np,Np,Nr,Nr,Nr);

    fmt2temp.update(T.rs_pq()*vhppp.rs_pq(), Nr, Nr, Nr, Np);
    t2t2c.update_as_tur_pqs(fmt2temp.pqs_r()*T.s_pqr(), Np,Np,Np,Nr,Nr,Nr); //check this one later!!!!
    t2t2c.update_as_pqr_stu(t2t2c.pqr_stu()-t2t2c.rqp_stu()-t2t2c.rpq_stu()-t2t2c.rpq_tsu()+t2t2c.qpr_stu()+t2t2c.qrp_tsu()-t2t2c.qrp_ust()+t2t2c.prq_tsu()+t2t2c.pqr_ust(), Np, Np, Np, Nr, Nr, Nr);

    fmt2temp.update(T.pq_rs()*vhhhp.pq_sr(), Np, Np, Np, Nr);
    t2t2d.update_as_qru_pst(fmt2temp.pqs_r()*T.q_prs(), Np,Np,Np,Nr,Nr,Nr);
    t2t2d.update_as_pqr_stu(t2t2d.pqr_stu()-t2t2d.qpr_stu()-t2t2d.rpq_stu()-t2t2d.rpq_uts()+t2t2d.prq_stu()+t2t2d.qrp_uts()-t2t2d.qrp_ust()+t2t2d.rqp_uts()+t2t2d.pqr_ust(), Np, Np, Np, Nr, Nr, Nr);

    if(timing){
        cout << "Quadratic t2 terms:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }

    // #########################
    // ##  Updating t3        ##
    // #########################

    sp_mat t3_temp = t2a.pqr_stu() - t2b.pqr_stu(); //CCDT-1 contribution

    t3_temp += t2t3a.pqr_stu();
    t3_temp -= .5*t2t3b.pqr_stu();
    t3_temp -= .5*t2t3c.pqr_stu();
    t3_temp -= .5*t2t3d.pqr_stu();
    t3_temp -= .5*t2t3e.pqr_stu();
    t3_temp += .25*t2t3f.pqr_stu();
    t3_temp += .25*t2t3g.pqr_stu();

    t3_temp += t2t2b.pqr_stu();
    t3_temp -= .5*t2t2c.pqr_stu(); // (*)
    t3_temp += .5*t2t2d.pqr_stu();

    t3_temp += .5*t3a.pqr_stu();
    t3_temp += .5*t3b.pqr_stu();
    t3_temp += t3c.pqr_stu();      // (*)

    //Some terms (*) have been errorprone at earlier stages of the code. They should work in the current version.

    T3.update_as_pqr_stu(t3_temp, Np,Np,Np,Nr,Nr,Nr);
    T3.set_amplitudes(ebs.vEnergy); //divide by energy denominator

    if(timing){
        cout << "Updating t3:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }

    //Calculating the triples contributions to T2
    fmD10b.update_as_q_rsp(vphpp.p_qrs()*T3.uqr_stp(), Np,Np,Nr,Nr);
    fmD10b.update(fmD10b.pq_rs() - fmD10b.qp_rs(), Np, Nq, Nr, Ns);

    fmD10c.update_as_pqr_s(T3.pqs_tur()*vhhhp.pqs_r(), Np,Np,Nr,Nr); //remember to permute these
    fmD10c.update(fmD10c.pq_rs() - fmD10c.pq_sr(), Np,Np,Nr,Nr);

    // #########################
    // ##  Updating t2        ##
    // #########################

    Tprev.update(T.pq_rs(), Np,Nq,Nr,Ns); //When using relaxation we need to store the previous amplitudes

    T.update(vpphh.pq_rs() + .5*(L1 + L2) + L3 + .25*Q1 + Q2 - .5*Q3 - .5*Q4 - .5*(fmD10b.pq_rs() - fmD10c.pq_rs()), Np, Nq, Nr, Ns);
    T.set_amplitudes(ebs.vEnergy); //divide updated amplitides by energy denominator
    T.update(alpha*Tprev.pq_rs() + (1.0-alpha)*T.pq_rs(), Np, Nq,Nr,Ns);

    energy(); //Calculate the energy
    T.shed_zeros();
    T.map_indices();
    if(timing){
        cout << "Updating t2:" << tm-omp_get_wtime() << endl;
        tm = omp_get_wtime();
    }
}


void ccdt_mp::energy(){
    // ##################################################
    // ##                                              ##
    // ## Calculate Correlation Energy                 ##
    // ##                                              ##
    // ##################################################

    sp_mat cv = vhhpp.pq_rs() * T.pq_rs();
    mat Cv(cv); //this is inefficient: does not utilize sp_mat functionality, one possibility through unpack_sp_mat
    double C_ = 0;
    for(int i = 0; i<Cv.n_cols; i++){
        C_+= Cv(i,i);
    }

    correlation_energy = .25*C_;
    cout << "[CCDT]["  << iterations  << "]" << "Energy               :" << .25*C_ << endl;
    cout << "[CCDT]["  << iterations  << "]" << "Energy (per particle):" << .25*C_/iSetup.iNh << endl;

}
