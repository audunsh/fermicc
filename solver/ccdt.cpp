#include "ccdt.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "solver/flexmat.h"
#include "basis/electrongas.h"
#include "solver/initializer.h"
#include "solver/unpack_sp_mat.h"
#include <time.h>

//#include <eigen/Eigen/Dense>
//#include <eigen/Eigen/Sparse>

using namespace std;
using namespace arma;

ccdt::ccdt(electrongas bs){
    ebs = bs;
    iSetup = initializer(bs);

    //setup all interaction matrices
    clock_t t;
    t =  clock();
    iSetup.sVhhhhO();
    cout << "Vhhhh init time:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
    t = clock();


    //the triples specific interactions
    iSetup.sVppppBlock();
    iSetup.sVhpppBlock();
    iSetup.sVppphBlock();

    iSetup.sVhphh();
    iSetup.sVphhp();




    //iSetup.sVppppO();  //DISABLE THIS ONE




    cout << "Vpppp init time:" <<  (float)(clock()- t)/CLOCKS_PER_SEC << endl;
    t = clock();

    iSetup.sVhhpp();
    cout << "Vhhpp init time:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
    t = clock();

    //iSetup.sVpphh();
    cout << "Vpphh init time:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
    t = clock();

    iSetup.sVhpph();
    cout << "Vhpph init time:" <<  (float)(clock()-t)/CLOCKS_PER_SEC << endl;

    //convert interaction data to flexmat objects
    vhhhh.init(iSetup.vValsVhhhh, iSetup.iVhhhh, iSetup.jVhhhh, iSetup.kVhhhh, iSetup.lVhhhh, iSetup.iNh, iSetup.iNh, iSetup.iNh, iSetup.iNh);
    vhhhh.shed_zeros();


    //vpppp.init(iSetup.vValsVpppp, iSetup.aVpppp, iSetup.bVpppp, iSetup.cVpppp, iSetup.dVpppp, iSetup.iNp, iSetup.iNp, iSetup.iNp, iSetup.iNp);
    //vpppp.shed_zeros();


    vhpph.init(iSetup.vValsVhpph, iSetup.iVhpph, iSetup.aVhpph, iSetup.bVhpph, iSetup.jVhpph, iSetup.iNh, iSetup.iNp, iSetup.iNp, iSetup.iNh);
    vhpph.shed_zeros();
    vhhpp.init(iSetup.vValsVhhpp, iSetup.iVhhpp, iSetup.jVhhpp, iSetup.aVhhpp, iSetup.bVhhpp, iSetup.iNh, iSetup.iNh, iSetup.iNp, iSetup.iNp);
    vhhpp.shed_zeros();
    vpphh.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    vpphh.shed_zeros();

    //mat H(vpphh.pq_rs());
    //H.save("pp_v_hh2.txt", raw_ascii);


    //set up first T2-amplitudes
    T.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    T.shed_zeros();
    T.set_amplitudes(bs.vEnergy);
    t = clock();
    T.map_indices();
    cout << "Amplitude mapping time:" <<  (float)(clock()-t)/CLOCKS_PER_SEC << endl;

    check_matrix_consistency();



    energy();

    for(int i = 0; i < 25; i++){
        //advance_intermediates();
        //cout << i+1 << " ";
        advance();

    }
    //cout << CCSD_SG_energy() << endl;
    cout << "Energy per electron:" << correlation_energy/iSetup.iNh << endl;


}


sp_mat ccdt::t2t2_block_multiplication(sp_mat spT1, blockmat bmV, sp_mat spT2){
    int iNp = iSetup.iNp;
    int iNh = iSetup.iNh;

    sp_mat ret;
    ret.set_size(iNp*iNp*iNp, iNh*iNh*iNh);

    int N = bmV.uN;

    field<uvec> stream;

    for(int i = 0; i < N; ++i){
        //retrieve block data
        stream = bmV.get_block(i);
    }
}



void ccdt::check_matrix_consistency(){
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

void ccdt::L1_dense_multiplication(){
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

void ccdt::L1_block_multiplication(){
    //perform Vpppp.pq_rs()*T.pq_Rs() using the block scheme

    uint N = iSetup.bmVpppp.uN;
    int Np = iSetup.iNp;
    int Nh = iSetup.iNh;
    int Np2 = Np*Np;
    L1.clear();
    L1.set_size(Np2,Np2);
    //L1 *= 0;
    field<uvec> stream;
    //uvec nx, ny;
    vec vals;
    umat coo;
    uint a,b,c,d, Na, mm,ab,cd;
    //uvec ab,ba,bc,cb, Na, row, col;
    sp_mat L1part(Np2, Np2);
    sp_mat Ttemp(Np2, Nh*Nh);
    double val;



    for(uint i = 0; i < N; ++i){

        coo.clear();
        vals.clear();

        L1part.clear();
        L1part.set_size(Np2,Np2);

        stream = iSetup.bmVpppp.get_block(i);

        uint Na = stream(0).size(); //is the usage of uint acceptable for these sized matrices?
        coo.set_size(2,Na*Na);
        vals.set_size(Na*Na);
        mm = 0;

        for(uint p = 0; p<Na; ++p){
            a = stream(0)(p);
            b = stream(1)(p);
            ab = a + b*Np;
            //Interaction below has already passed d(k_p+k_q, k_r + k_s) && m_p==m_r && m_q == m_s
            val = iSetup.bs.v2(a+Nh,b+Nh,a+Nh,b+Nh);
            coo(0,mm) = ab;
            coo(1,mm) = ab;
            vals(mm) = val; //iSetup.bs.v2(a+Nh,b+Nh,a+Nh,b+Nh);
            mm+=1;

            //NOTE: Actually going through the kroenecker deltas here, possible to skip many tests in the interaction p==r, q==s
            //Interaction below has already passed d(k_p+k_q, k_r + k_s)

            for(uint q = p+1; q<Na; ++q){
                //more symmetries to utilize here (spin)
                c = stream(2)(q);
                d = stream(3)(q);
                val = iSetup.bs.v2(a+Nh,b+Nh,c+Nh,d+Nh); //create separate function here

                cd = c + d* Np;
                coo(0,mm) = ab;
                coo(1,mm) = cd;
                vals(mm) = val;
                mm+=1;
                coo(0,mm) = cd;
                coo(1,mm) = ab;
                vals(mm) = val;
                mm+=1;
            }

        }


        L1part = sp_mat(coo, vals, Np2,Np2); //this is what slows down the calculation: is it possible to rearrange the elements somehow? (in column increasing order?)
        Ttemp = T.rows(stream(4)); //load only elements in row
        L1 += L1part*Ttemp;

    }


}


void ccdt::advance_intermediates(){
    int Na = iSetup.iNp;
    int Nb = iSetup.iNp;
    int Nc = iSetup.iNp;
    int Ni = iSetup.iNh;
    int Nj = iSetup.iNh;
    int Nk = iSetup.iNh;


    //Doubles contributions to t2

    sp_mat spI1 = vhhhh.pq_rs() + .5*vhhpp.pq_rs()*T.pq_rs();
    fmI2temp.update(.5*vhhpp.pr_qs()*T.rp_qs(), Na,Nb,Ni,Nj);
    fmI2.update(vhpph.pq_rs() + fmI2temp.pq_rs(),Na,Nb,Ni,Nj);

    sp_mat I3temp = vhhpp.p_rsq()*T.pqs_r();
    sp_mat I4temp = T.p_rsq()*vhhpp.pqs_r();

    L1_dense_multiplication();

    L2 = .5*T.pq_rs()*spI1;
    fmL3.update(T.pr_qs()*fmI2.rp_qs(), Na,Nb,Ni,Nj);
    L3 = fmL3.pr_qs()-fmL3.pr_sq()-fmL3.rp_qs()+fmL3.rp_sq();

    fmQ2.update_as_pqr_s(.5*T.pqr_s()*I3temp, Na,Nb,Ni,Nj);
    Q2 = fmQ2.pq_rs()-fmQ2.pq_sr();

    fmQ3.update_as_q_prs(.5*I4temp*T.q_prs(),Na,Nb,Ni,Nj);
    Q3 = fmQ3.pq_rs()-fmQ3.qp_rs();

    //Triples contributions to t2

    fmD10b.update_as_q_rsp(vphpp.p_qrs()*T3.uqr_stp(), Na,Nb,Ni,Nj);
    fmD10c.update_as_pqr_s(T3.pqs_tur()*vhhhp.pqs_r(), Na,Nb,Ni,Nj); //remember to permute these

    //Calculating the t3 amplitudes
    t2t3a.update_as_qtru_ps(T3.qtru_sp()*vhhpp.qs_pr()*T.sq_pr(), Na,Nb,Nc,Ni,Nj,Nk );
    t2t3b.update_as_pqtru_s(T3.pqtru_s()*vhhpp.q_prs()*T.rpq_s(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3c.update_as_sqtru_p(T3.sqtru_p()*vhhpp.s_pqr()*T.rsp_q(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3d.update_as_qru_pst(T3.pru_stq()*vhhpp.pqs_r()*T.q_prs(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3e.update_as_tru_pqs(T3.sru_tpq()*vhhpp.qrs_p()*T.s_pqr(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3f.update_as_pqru_st(T3.pqru_st()*vhhpp.pq_rs()*T.pq_rs(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3g.update_as_stru_pq(T3.stru_pq()*vhhpp.rs_pq()*T.rs_pq(), Na,Nb,Nc,Ni,Nj,Nk);



    t2t2b.update_as_psq_rtu(T.pr_sq()*vhppp.pr_qs()*T.p_qrs(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t2c.update_as_tur_pqs(T.rs_pq()*vhppp.rs_pq()*T.s_pqr(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t2d.update_as_qru_pst(T.pq_rs()*vhhph.pq_rs()*T.q_prs(), Na,Nb,Nc,Ni,Nj,Nk);

    //t3a.update_as_pq_rstu(vpppp.pq_rs()∗T3.pq_rstu()); //Note that this will probably be replaced by a block implementation.



    t3b.update_as_pqru_st(T3.pqrs_tu()*vphhp.pq_rs(), Na,Nb,Nc,Ni,Nj,Nk);

    t3c.update_as_ps_qrtu(vphhp.pr_qs()*T3.sp_qrtu(), Na,Nb,Nc,Ni,Nj,Nk);

    t2a.update_as_qru_pst(vppph.pqs_r()*T.q_prs(), Na,Nb,Nc,Ni,Nj,Nk);

    t2b.update_as_pqs_rtu(T.pqr_s()*vhphh.p_qrs(), Na,Nb,Nc,Ni,Nj,Nk);




    //update t2
    T.update(vpphh.pq_rs() + .5*L1+L2+L3-Q2-Q3, Na, Nb, Ni, Nj);
    T.set_amplitudes(ebs.vEnergy); //divide updated amplitides by energy denominator

    //update t3


    energy();


}

void ccdt::advance(){
    //advance the solution one step
    int Na = iSetup.iNp;
    int Nb = iSetup.iNp;
    int Nc = iSetup.iNp;

    int Ni = iSetup.iNh;
    int Nj = iSetup.iNh;
    int Nk = iSetup.iNh;
    bool timing = false; //time each contribution calculation and print to screen (each iteration)
    clock_t t;

    if(timing){t = clock();}
    //L1 = vpppp.pq_rs()*T.pq_rs();
    //L1_block_multiplication();
    L1_dense_multiplication();
    if(timing){
        cout << "Time spent on L1:" << (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    //make a little section here and insert block diagonal replacement


    L2 = T.pq_rs()*vhhhh.pq_rs();
    if(timing){
        cout << "Time spent on L2:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    fmL3.update(vhpph.sq_rp()*T.qs_pr(),  Na,Nb,Ni,Nj);
    L3 = fmL3.rq_sp() - fmL3.qr_sp() -fmL3.rq_ps() +fmL3.qr_ps();
    if(timing){
        cout << "Time spent on L3:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    fmQ1.update(T.rs_pq()*vhhpp.rs_pq()*T.rs_pq(),  Na,Nb,Ni,Nj);
    Q1 = fmQ1.rs_pq();
    if(timing){
        cout << "Time spent on Q1:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    fmQ2.update(T.pr_qs()*vhhpp.rp_qs()*T.sq_pr(),  Na,Nb,Ni,Nj); //needs realignment and permutations
    Q2 = fmQ2.pr_qs()-fmQ2.pr_sq(); //permuting elements
    if(timing){
        cout << "Time spent on Q2:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    fmQ3.update_as_r_pqs((T.r_sqp()*vhhpp.prs_q())*T.r_pqs(),  Na,Nb,Ni,Nj); //needs realignment and permutations
    Q3 = fmQ3.pq_rs() - fmQ3.pq_sr(); //permuting elements
    if(timing){
        cout << "Time spent on Q3:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    fmQ4.update_as_p_qrs(T.p_srq()*vhhpp.pqr_s()*T.p_qrs(),  Na,Nb,Ni,Nj); //needs realignment and permutations
    Q4 = fmQ4.pq_rs() - fmQ4.qp_rs(); //permuting elements
    if(timing){
        cout << "Time spent on Q4:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    //Triples contributions to t2
    fmD10b.update_as_q_rsp(vphpp.p_qrs()*T3.uqr_stp(), Na,Nb,Ni,Nj);
    fmD10b.pq_rs() - fmD10b.qp_rs();
    fmD10c.update_as_pqr_s(T3.pqs_tur()*vhhhp.pqs_r(), Na,Nb,Ni,Nj); //remember to permute these
    fmD10c.pq_rs() - fmD10c.pq_sr();

    //Calculating the t3 amplitudes



    t2t3a.update_as_qtru_ps(T3.qtru_sp()*vhhpp.qs_pr()*T.sq_pr(), Na,Nb,Nc,Ni,Nj,Nk );
    //t2t3a.pqr_stu()-t2t3a.qpr_stu()-t2t3a.rpq_stu()-t2t3a.rpq_tsu()+t2t3a.prq_stu()+t2t3a.qrp_tsu()-t2t3a.qrp_ust()+t2t3a.rqp_tsu()+t2t3a.pqr_ust();

    t2t3b.update_as_pqtru_s(T3.pqtru_s()*vhhpp.q_prs()*T.rpq_s(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3b.pqr_stu()-t2t3b.pqr_tsu()-t2t3b.pqr_ust();


    t2t3c.update_as_sqtru_p(T3.sqtru_p()*vhhpp.s_pqr()*T.rsp_q(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3c.pqr_stu()-t2t3c.qpr_stu()-t2t3c.rpq_stu();


    t2t3d.update_as_qru_pst(T3.pru_stq()*vhhpp.pqs_r()*T.q_prs(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3d.pqr_stu()-t2t3d.qpr_stu()-t2t3d.rpq_stu()-t2t3d.rpq_sut()+t2t3d.prq_stu()+t2t3d.qrp_sut()-t2t3d.qrp_tus()+t2t3d.rqp_sut()+t2t3d.pqr_tus();


    t2t3e.update_as_tru_pqs(T3.sru_tpq()*vhhpp.qrs_p()*T.s_pqr(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3e.pqr_stu()-t2t3e.rqp_stu()-t2t3e.rpq_stu()-t2t3e.rpq_tsu()+t2t3e.qpr_stu()+t2t3e.qrp_tsu()-t2t3e.qrp_ust()+t2t3e.prq_tsu()+t2t3e.pqr_ust();


    t2t3f.update_as_pqru_st(T3.pqru_st()*vhhpp.pq_rs()*T.pq_rs(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3f.pqr_stu()-t2t3f.pqr_uts()-t2t3f.pqr_ust();


    t2t3g.update_as_stru_pq(T3.stru_pq()*vhhpp.rs_pq()*T.rs_pq(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t3g.pqr_stu()-t2t3g.rqp_stu()-t2t3g.rpq_stu();


    //Special attention needed here:

    t2t2b.update_as_psq_rtu(T.pr_sq()*vhppp.pr_qs()*T.p_qrs(), Na,Nb,Nc,Ni,Nj,Nk);
    //(T.pr_sq()*vhppp.pr_qs())

    //t2t2b.pqr_stu()-t2t2b.qpr_stu()-t2t2b.rpq_stu()-t2t2b.rpq_tsu()+t2t2b.prq_stu()+t2t2b.qrp_tsu()-t2t2b.qrp_ust()+t2t2b.rqp_tsu()+t2t2b.pqr_ust()


    t2t2c.update_as_tur_pqs(T.rs_pq()*vhppp.rs_pq()*T.s_pqr(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t2c.pqr_stu()-t2t2c.rqp_stu()-t2t2c.rpq_stu()-t2t2c.rpq_tsu()+t2t2c.qpr_stu()+t2t2c.qrp_tsu()-t2t2c.qrp_ust()+t2t2c.prq_tsu()+t2t2c.pqr_ust();


    t2t2d.update_as_qru_pst(T.pq_rs()*vhhph.pq_rs()*T.q_prs(), Na,Nb,Nc,Ni,Nj,Nk);
    t2t2d.pqr_stu()-t2t2d.qpr_stu()-t2t2d.rpq_stu()-t2t2d.rpq_uts()+t2t2d.prq_stu()+t2t2d.qrp_uts()-t2t2d.qrp_ust()+t2t2d.rqp_uts()+t2t2d.pqr_ust();


    //t3a.update_as_pq_rstu(vpppp.pq_rs()∗T3.pq_rstu()); //Note that this will probably be replaced by a block implementation.
    //t3a.pqr_stu()-t3a.rqp_stu()-t3a.rpq_stu()



    t3b.update_as_pqru_st(T3.pqrs_tu()*vphhp.pq_rs(), Na,Nb,Nc,Ni,Nj,Nk);
    t3b.pqr_stu()-t3b.pqr_sut()-t3b.pqr_tus();


    t3c.update_as_ps_qrtu(vphhp.pr_qs()*T3.sp_qrtu(), Na,Nb,Nc,Ni,Nj,Nk);
    t3c.pqr_stu()-t3c.qpr_stu()-t3c.rpq_stu()-t3c.rpq_tsu()+t3c.prq_stu()+t3c.qrp_tsu()-t3c.qrp_ust()+t3c.rqp_tsu()+t3c.pqr_ust();


    t2a.update_as_qru_pst(vppph.pqs_r()*T.q_prs(), Na,Nb,Nc,Ni,Nj,Nk);
    t2a.pqr_stu()-t2a.qpr_stu()-t2a.rpq_stu()-t2a.rpq_uts()+t2a.prq_stu()+t2a.qrp_uts()-t2a.qrp_ust()+t2a.rqp_uts()+t2a.pqr_ust();


    t2b.update_as_pqs_rtu(T.pqr_s()*vhphh.p_qrs(), Na,Nb,Nc,Ni,Nj,Nk);
    t2b.pqr_stu()-t2b.rqp_stu()-t2b.rpq_stu()-t2b.rpq_tsu()+t2b.qpr_stu()+t2b.qrp_tsu()-t2b.qrp_ust()+t2b.prq_tsu()+t2b.pqr_ust();





    //update t2
    T.update(vpphh.pq_rs() + .5*L1+L2+L3-Q2-Q3, Na, Nb, Ni, Nj);
    T.set_amplitudes(ebs.vEnergy); //divide updated amplitides by energy denominator
    T.shed_zeros();
    //update t3


    energy();
}


double ccdt::CCSD_SG_energy(){
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


void ccdt::energy(){
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
    cout << "(CCD)Energy:" << .25*C_ << endl;
    cout << "(CCD)Energy (per particle):" << .25*C_/iSetup.iNh << endl;

}
