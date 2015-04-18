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
    clock_t t;
    t =  clock();
    iSetup.sVhhhhO();
    cout << "Vhhhh init time:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
    t = clock();

    iSetup.sVppppBlock();

    //iSetup.sVppppO();
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
    t = clock();

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


    //set up first T2-amplitudes
    T.init(iSetup.vValsVpphh, iSetup.aVpphh, iSetup.bVpphh, iSetup.iVpphh, iSetup.jVpphh, iSetup.iNp, iSetup.iNp, iSetup.iNh, iSetup.iNh);
    T.set_amplitudes(bs.vEnergy);
    t = clock();
    T.map_indices();
    cout << "Amplitude mapping time:" <<  (float)(clock()-t)/CLOCKS_PER_SEC << endl;

    // HOW TO SET UP FLEXMAT OBJECTS FROM CSC-MATRICES
    // flexmat V1;
    // V1.update(vhhhh.pq_rs(),vhhhh.iNp, vhhhh.iNq, vhhhh.iNr, vhhhh.iNs); //update (or initialize) with an sp_mat object (requires unpacking)


    //compare matrix multiplication schemes (block vs. sparse)
    t = clock();
    L2 =  vpppp.pq_rs()*T.pq_rs();
    cout << "Sparse mult time:" <<  (float)(clock()-t)/CLOCKS_PER_SEC << endl;

    t = clock();
    L1_block_multiplication();
    cout << "Blocked/sparse mult time:" <<  (float)(clock()-t)/CLOCKS_PER_SEC << endl;

    //compare L1, L2

    /*
    for(int a = 0; a < iSetup.iNp; a++){
        for(int b = 0; b < iSetup.iNp; b++){
            for(int c = 0; c < iSetup.iNh; c++){
                for(int d = 0; d < iSetup.iNh; d++){
                    if(L2(a + b*iSetup.iNp, c + d*iSetup.iNh) != L1(a + b*iSetup.iNp, c + d*iSetup.iNh)){
                        cout << L2(a + b*iSetup.iNp, c + d*iSetup.iNh) << " " << L1(a + b*iSetup.iNp, c + d*iSetup.iNh) << endl;
                    }
                }
            }
        }
    }
    */


    /*
    cout << L2.n_nonzero << " " << L1.n_nonzero << endl;

    for(int i= 0; i<L2.n_nonzero; ++i){
        if(L2.values[i] != L1.values[i]){
            cout << L2.values[i]<< " " << L1.values[i] << endl;
        }
    }
    */







    energy();

    for(int i = 0; i < 25; i++){
        //advance_intermediates();
        advance();
    }


}

void ccd::L1_block_multiplication(){
    //perform Vpppp.pq_rs()*T.pq_Rs() using the block scheme
    clock_t t0, ti,tm,ts;

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
    ti = 0;
    tm = 0;
    for(uint i = 0; i < N; ++i){
        t0 = clock();
        coo.clear();
        vals.clear();

        L1part.clear();
        L1part.set_size(Np2,Np2);

        stream = iSetup.bmVpppp.get_block(i);
        //row = stream(0) + Np*stream(1);
        //col = stream(2) + Np*stream(3);
        uint Na = stream(0).size();
        coo.set_size(Na*Na,2);
        vals.set_size(Na*Na);
        mm = 0;
        for(int p = 0; p<Na; ++p){
            a = stream(0)(p);
            b = stream(1)(p);
            ab = a + b*Np;
            val = iSetup.bs.v2(a+Nh,b+Nh,a+Nh,b+Nh);
            //L1part(ab,ab) = val;


            coo.col(0)(mm) = ab;
            coo.col(1)(mm) = ab;
            vals(mm) = val;
            mm+=1;


            for(int q = p+1; q<Na; ++q){
                //more symmetries to utilize here

                c = stream(2)(q);
                d = stream(3)(q);
                val = iSetup.bs.v2(a+Nh,b+Nh,c+Nh,d+Nh);

                cd = c + d* Np;
                //L1part(ab,cd) = val;  //this is inefficient, as we need to perform lookups in the compressed sp_mat object
                //L1part(cd,ab) = val;



                coo.col(0)(mm) = ab;
                coo.col(1)(mm) = cd;
                vals(mm) = val;
                mm+=1;

                coo.col(0)(mm) = cd;
                coo.col(1)(mm) = ab;
                vals(mm) = val;
                mm+=1;


                /*
                L1part(a+b*Np,d+c*Np) = -val;
                L1part(d+c*Np,a+b*Np) = -val;

                L1part(b+a*Np,c+d*Np) = -val;
                L1part(c+d*Np,b+a*Np) = -val;

                L1part(b+a*Np,d+c*Np) = val;
                L1part(d+c*Np,b+a*Np) = val;
                */




            }
        }

        //O = ones(Na);
        //a = convert_to<uvec>::from(kron(O, stream(0)));
        //values = iSetup.V3(stream(0), stream(1), stream(2), stream(3));
        //unfold vectors and symmetries
        //locations.set_size(values.size(), 2);
        //locations.col(0) = stream(0) + Np*stream(1);
        //locations.col(1) = stream(2) + Np*stream(3);
        L1part = sp_mat(coo.t(), vals, Np2,Np2);
        ts += (clock()-t0);
        t0 = clock();
        Ttemp = T.rows(stream(4));
        ti += (clock()-t0);
        t0 = clock();
        L1 += L1part*Ttemp;
        tm += (clock()-t0);

    }
    cout << "Setup time:" <<  (float)(ti)/CLOCKS_PER_SEC << endl;
    cout << "Init  time:" <<  (float)(ti)/CLOCKS_PER_SEC << endl;
    cout << "Mult  time:" <<  (float)(tm)/CLOCKS_PER_SEC << endl;
    //L1 *= .5;

}


void ccd::advance_intermediates(){
    int Np = iSetup.iNp;
    int Nq = iSetup.iNp;
    int Nr = iSetup.iNh;
    int Ns = iSetup.iNh;

    //fmI1.update(vhhhh.pq_rs() + .5*vhhpp.pq_rs()*T.pq_rs(), Np, Nq, Nr, Ns);

    sp_mat spI1 = vhhhh.pq_rs() + .5*vhhpp.pq_rs()*T.pq_rs();
    fmI2temp.update(.5*vhhpp.pr_qs()*T.rp_qs(), Np, Nr, Nq, Ns);
    fmI2.update(vhpph.pq_rs() + fmI2temp.pq_rs(), Nr, Np, Nq, Ns);

    sp_mat I3temp = vhhpp.p_rsq()*T.pqs_r();
    sp_mat I4temp = T.p_rsq()*vhhpp.pqs_r();


    L1 = .5*vpppp.pq_rs()*T.pq_rs();
    L2 = .5*T.pq_rs()*spI1;
    fmL3.update(T.pr_qs()*fmI2.rp_qs(), Np, Nr, Nq, Ns);
    L3 = fmL3.pr_qs()-fmL3.pr_sq()-fmL3.rp_qs()+fmL3.rp_sq();

    fmQ2.update_as_pqr_s(.5*T.pqr_s()*I3temp, Np, Nq, Nr, Ns);
    Q2 = fmQ2.pq_rs()-fmQ2.pq_sr();

    fmQ3.update_as_q_prs(.5*I4temp*T.q_prs(), Np, Nq, Nr, Ns);
    Q3 = fmQ3.pq_rs()-fmQ3.qp_rs();

    T.update(vpphh.pq_rs() + L1+L2+L3-Q2-Q3, Np, Nq, Nr, Ns);
    T.set_amplitudes(ebs.vEnergy); //divide updated amplitides by energy denominator

    energy();


}

void ccd::advance(){
    //advance the solution one step
    int Np = iSetup.iNp;
    int Nq = iSetup.iNp;
    int Nr = iSetup.iNh;
    int Ns = iSetup.iNh;
    bool timing = true; //time each contribution calculation and print to screen (each iteration)
    clock_t t;

    if(timing){t = clock();}
    //L1 = vpppp.pq_rs()*T.pq_rs();
    L1_block_multiplication();
    if(timing){
        cout << "Time spent on L1:" << (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    //make a little section here and insert block diagonal replacement


    L2 = T.pq_rs()*vhhhh.pq_rs();
    if(timing){
        cout << "Time spent on L2:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    fmL3.update(vhpph.sq_rp()*T.qs_pr(), Ns, Nq, Np, Nr);
    L3 = fmL3.rq_sp() - fmL3.qr_sp() - fmL3.rq_ps() + fmL3.qr_ps();
    if(timing){
        cout << "Time spent on L3:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    fmQ1.update(T.rs_pq()*vhhpp.rs_pq()*T.rs_pq(), Nr, Ns, Np,Nq);
    Q1 = fmQ1.rs_pq();
    if(timing){
        cout << "Time spent on Q1:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    fmQ2.update(T.pr_qs()*vhhpp.rp_qs()*T.sq_pr(), Np, Nr, Nq, Ns); //needs realignment and permutations
    Q2 = fmQ2.pr_qs()-fmQ2.pr_sq(); //permuting elements
    if(timing){
        cout << "Time spent on Q2:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    fmQ3.update_as_r_pqs((T.r_sqp()*vhhpp.prs_q())*T.r_pqs(), Np, Nq, Nr, Ns); //needs realignment and permutations
    Q3 = fmQ3.pq_rs() - fmQ3.pq_sr(); //permuting elements
    if(timing){
        cout << "Time spent on Q3:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    fmQ4.update_as_p_qrs(T.p_srq()*vhhpp.pqr_s()*T.p_qrs(), Np, Nq, Nr, Ns); //needs realignment and permutations
    Q4 = fmQ4.pq_rs() - fmQ4.qp_rs(); //permuting elements
    if(timing){
        cout << "Time spent on Q4:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    T.update(vpphh.pq_rs() + .5*(L1 + L2) + L3 + .25*Q1 + Q2 - .5*Q3 - .5*Q4, Np, Nq, Nr, Ns);
    T.set_amplitudes(ebs.vEnergy); //divide updated amplitides by energy denominator
    if(timing){
        cout << "Time spent on T:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}

    energy();
    if(timing){
        cout << "Time spent on e:" <<  (clock() - (float)t)/CLOCKS_PER_SEC << endl;
        t = clock();}
    T.map_indices();
}


double ccd::CCSD_SG_energy(){
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


void ccd::energy(){
    //Calculate the ground state energy
    sp_mat cv = vhhpp.pq_rs() * T.pq_rs();
    mat Cv(cv); //this is inefficient: does not utilize sp_mat functionality, one possibility through unpack_sp_mat
    double C_ = 0;
    for(int i = 0; i<Cv.n_cols; i++){
        C_+= Cv(i,i);
    }
    cout << "(CCD)Energy:" << .25*C_ << endl;
}
