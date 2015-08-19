#include "ccd_mp.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "solver/flexmat.h"
#include "basis/electrongas.h"
#include "solver/initializer.h"
#include "solver/unpack_sp_mat.h"
#include <time.h>
#include <omp.h>

//#include <eigen/Eigen/Dense>
//#include <eigen/Eigen/Sparse>

using namespace std;
using namespace arma;

ccd_mp::ccd_mp(electrongas bs, double a){

    // ##################################################
    // ##                                              ##
    // ## CCD, initialization                          ##
    // ##                                              ##
    // ##################################################

    alpha = a; //relaxation parameter
    iterations = 0; //current number of iterations
    ebs = bs;
    iSetup = initializer(bs);
    double tm = omp_get_wtime();

    //setup all interaction matrices
    clock_t t;
    t = clock();

    iSetup.sVppppBlock_mp();
    #pragma omp parallel  num_threads(3)
    {
    if(omp_get_thread_num()==0){
    iSetup.sVhhhhO();
    //iSetup.sVppppBlock();
    }
    if(omp_get_thread_num()==1){
    iSetup.sVhhpp();
    }
    if(omp_get_thread_num()==2){
    iSetup.sVhpph();
    }
    }
    //convert interaction data to flexmat objects
    cout << "Time spent in parallell:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
    t = clock();
    vhhhh.init(iSetup.vValsVhhhh, iSetup.iVhhhh, iSetup.jVhhhh, iSetup.kVhhhh, iSetup.lVhhhh, iSetup.iNh, iSetup.iNh, iSetup.iNh, iSetup.iNh);
    vhhhh.shed_zeros();
    vhpph.init(iSetup.vValsVhpph, iSetup.iVhpph, iSetup.aVhpph, iSetup.bVhpph, iSetup.jVhpph, iSetup.iNh, iSetup.iNp, iSetup.iNp, iSetup.iNh);
    vhpph.shed_zeros();
    vhhpp.init(iSetup.vValsVhhpp, iSetup.iVhhpp, iSetup.jVhhpp, iSetup.aVhhpp, iSetup.bVhhpp, iSetup.iNh, iSetup.iNh, iSetup.iNp, iSetup.iNp);
    vhhpp.shed_zeros();
    vpphh.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    vpphh.shed_zeros();

    cout << "Time spent in serial:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
    t = clock();
    //set up first T2-amplitudes
    T.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    T.shed_zeros();
    T.set_amplitudes(bs.vEnergy);
    t = clock();
    T.map_indices();

    cout << "[CCD_mp]Real world initialization time:" << omp_get_wtime()-tm << endl;

    energy();


    for(int i = 0; i < 25; i++){
        //advance_intermediates();
        //cout << i+1 << " ";
        iterations += 1;
        advance();

    }
}

void ccd_mp::check_matrix_consistency(){
    // ##################################################
    // ##                                              ##
    // ## Matrix consistency test, debugging function  ##
    // ##                                              ##
    // ##################################################

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

void ccd_mp::L1_dense_multiplication(){
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


void ccd_mp::advance(){
    //advance the solution one step
    int Np = iSetup.iNp;
    int Nq = iSetup.iNp;
    int Nr = iSetup.iNh;
    int Ns = iSetup.iNh;
    //bool timing = false; //time each contribution calculation and print to screen (each iteration)
    //clock_t t;
    //t = clock();
    L1_dense_multiplication();
    //cout << "Time spent on L1:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;


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


    Tprev.update(T.pq_rs(), Np,Nq,Nr,Ns);

    T.update(vpphh.pq_rs() + .5*(L1 + L2) + L3 + .25*Q1 + Q2 - .5*Q3 - .5*Q4, Np, Nq, Nr, Ns);
    T.set_amplitudes(ebs.vHFEnergy); //divide updated amplitides by energy denominator
    T.update(alpha*Tprev.pq_rs() + (1.0-alpha)*T.pq_rs(), Np, Nq,Nr,Ns);

    energy();

    T.shed_zeros();
    T.map_indices();
}


double ccd_mp::CCSD_SG_energy(){
    //Return correlation energy
    //Optional implementation for comparison
    //Inefficiency warning; performs lookup in sparse matrix
    int Np = iSetup.iNp;
    int Nh = iSetup.iNh;
    double val1, val2, val3, v_ijab;
    int a,b,i,j;
    val1 = 0.0;
    val2 = 0.0;
    val3 = 0.0;
    for(a=0; a<Np; a++){
        for(i = 0; i<Nh; i++){
            //val1 += h(a,i)*t1(a,i);
            for(b = 0; b<Np; b++){
                for(j = 0; j<Nh; j++){
                    v_ijab = vhhpp.pq_rs()(i+j*Nh,a+b*Np);
                    val2 += v_ijab*T.pq_rs()(a+b*Np,i+j*Nh);
                    //val3 += v_ijab*t1(a,i)*t1(b,j);
                }
            }
        }
    }
    val1 += .25*val2 +.5*val3;
    return val1;
}


void ccd_mp::energy(){
    //Calculate the ground state energy
    sp_mat cv = vhhpp.pq_rs() * T.pq_rs();
    //mat vhhpp2(vhhpp.pq_rs());
    //mat tpphh2(T.pq_rs());
    //mat Cv = vhhpp2*tpphh2;
    mat Cv(cv); //this is inefficient: does not utilize sp_mat functionality, one possibility through unpack_sp_mat
    double C_ = 0;
    for(uint i = 0; i<Cv.n_cols; i++){
        C_+= Cv(i,i);
    }


    correlation_energy = .25*C_;
    cout << "["  << iterations  << "]" << "[CCD]Energy               :" << .25*C_ << endl;
    cout << "["  << iterations  << "]" << "[CCD]Energy (per particle):" << .25*C_/iSetup.iNh << endl;

}
