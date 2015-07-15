#define ARMA_64BIT_WORD
#include <armadillo>

#include <fstream>
#include <iomanip>
#include <time.h>
#include <string>
#include "basis/electrongas.h"
#include "solver/ccsolve.h"
#include "solver/ccd.h"
#include "solver/initializer.h"
#include "solver/flexmat.h"
#include "solver/ccd_pt.h"
#include "solver/ccdt.h"
#include "solver/ccdt_mp.h"
#include "solver/ccd_mp.h"
#include "solver/amplitude.h"
#include "solver/bccd.h"

using namespace std;
using namespace arma;


int main()
{
    //TODO LIST
    //1. Speed up initialization
    //2. Experiment with uniquely reduced t3amps in setup
    electrongas fgas;
    fgas.generate_state_list2(5.0,1.0, 14);

    //cout << "Energy per particle:" << fgas.eref(14)/14.0 << " (a.u)"  << endl;
    //cout << "[Main] Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    cout << "[Main]" << setprecision(8) << "Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;

    cout << "[Main]G.Baardsens results:" << 1.9434 << endl;


    uint Np = fgas.iNbstates-fgas.iNparticles; //conflicting notation here
    uint Nh = fgas.iNparticles;

    //vec amptest = zeros(Np*Np*Np*Nh*Nh*Nh);
    //ccd_pt solver(fgas, 0);

    bccd solver1(fgas);
    //solver.t2.blocklengths.print();



    //testing consistency of t2 permutative mapping


    /*
    amplitude t2a(fgas, 8, {Np, Np, Nh, Nh});
    t2a.make_t3();
    t2a.uiCurrent_block = 1;
    //t2a.map6({1,2,3},{4,5,6}); //ab ij (0)
    //t2a.map6({1,2,-4},{5,-3,6}); //ab ij (0)


    t2a.uiCurrent_block = 0;
    t2a.map_t3_permutations();
    t2a.map6({1,2,3}, {4,6,5}); //, t2.fvConfigs(5)); //for use in t2a (1)
    //t2a.map6({1,2,-4}, {-3,5,6}); //, t2.fvConfigs(6)); //for use in t2b (2)
    t2a.init_t3_amplitudes();

    for(uint i = 0; i <t2a.blocklengths(0); ++i){
        t2a.getraw(1,i).print();
        cout << endl;
        //cout << t2b.Pab(i) << endl;
        //t2b.getraw(0,i).print();
        cout << endl;

        t2a.getraw_permuted(0,i, 5).print();
        cout << endl;
    }
    */




    /*
    clock_t t = clock();
    amplitude t2b(fgas, 8, {Np, Np, Nh, Nh});
    t2b.make_t3();
    t2b.map_t3_permutations();
    //t2b.init_amplitudes();
    cout << "time spent initializing amplitude:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
    */


    //cout << t2a.blocklengths(0) << t2b.blocklengths(0) << endl;

    /*
    for(uint i = 0; i <t2a.blocklengths(0); ++i){
        t2a.getraw(1,i).print();
        cout << endl;
        //cout << t2b.Pab(i) << endl;
        //t2b.getraw(0,i).print();
        cout << endl;

        t2b.getraw_permuted(0,i, 3).print();
        cout << endl;
    }
    */




    /*
    for(uint i = 0; i < t2b.fvConfigs(0).n_rows; ++i){
        cout << t2b.fvConfigs(0)(i) << " " << t2a.fvConfigs(0)(i) << endl;
    }



    for(uint i = 0; i<t2b.uvElements.n_rows; ++i){
        cout << t2a.uvElements(i) << " " << t2b.uvElements(i) << endl;

        //t2b.from(t2b.uvElements(i)).print();
        //cout << endl;


        //t2a.from(t2a.uvElements(i)).print();
        //cout << endl;
    }
    */

    /*
    t2a.uvSize.print();
    t2b.uvSize.print();
    for(uint i =1; i < 100000; i*=2){
        cout << t2a.from(i) << " " << t2b.from(i) << endl;
    }*/









    /*
    umat a(3,3);
    a(0,0) = 0;
    a(1,0) = 1;
    a(2,0) = 2;

    a(0,1) = 3;
    a(1,1) = 4;
    a(2,1) = 5;
    a(0,2) = 6;
    a(1,2) = 7;
    a(2,2) = 8;

    uvec ind = {0,2,};
    umat b = a.rows(ind);
    b.print();
    cout << endl;
    a.print();
    */

    //blockmap t2;
    //t2.init(fgas, 2, {Np,Np,Nh,Nh});
    //t2.map({1,2},{3,4});

    //ccsolve solver2(fgas);
    //solver2.CCSD_SG(2);
    //ccd_pt solver(fgas, .5);


    /*
    clock_t t = clock();
    amplitude t3;
    t3.init(fgas, 2, {Np, Np, Np, Nh, Nh, Nh});
    t3.make_t3();
    t3.map_t3_permutations();
    cout << "0:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
    for(uint i = 0; i < t3.blocklengths(0); ++i){
        t3.getraw_permuted(0,i,0).print();
        cout << endl;
        t3.getraw_permuted(0,i,3).print();
        cout << endl;

        cout << endl;}*/


    /*
    t3.map6({1,2,3},{4,5,6});
    t3.map6({1,3},{4,5,-2,6});
    cout << t3.blocklengths(0) << endl;
    cout << t3.blocklengths(1) << endl;
    */


    /*
    for(uint i = 0; i < t3.blocklengths(0); ++i){
        t3.getraw(0,i).print();
        cout << endl;
        t3.getraw(1,i).print();
        cout << endl;
        cout << endl;
    }
    */

    //mat test = { {1,2,3}, {1,2,3} };
    //mat C = { {1, 3, 5}, {2, 4, 6} };
    //C.print();


    /*
    uint Np = fgas.iNbstates-fgas.iNparticles; //conflicting notation here
    uint Nh = fgas.iNparticles;
    amplitude t2(fgas, 8, {Np, Np, Nh, Nh});

    //next amplitude


    //Temporary amplitude storage for permutations
    amplitude t2temp = t2;
    t2temp.map({1,2},{3,4}); //ab ij (0)

    t2temp.map({2,1},{3,4}); //ba ij (1)
    t2temp.map({1,2},{4,3}); //ab ji (2)
    t2temp.map({2,1},{4,3}); //ba ji (3)

    //the input configurations
    t2temp.map({-4,2}, {-1,3}); //for use in L3 update (4)
    t2temp.map({1,-3}, {-2,4}); //for use ni Q2 update (5)
    t2temp.map({1,2,-4}, {3});  //for use in Q3 update (6)



    t2.map({1,2},{3,4}); //ab ij (0)
    t2.map({2,1},{3,4}); //ba ij (1)
    t2.map({1,2},{4,3}); //ab ji (2)
    t2.map({2,1},{4,3}); //ba ji (3)

    //the input configurations
    t2.map({-4,2}, {-1,3}); //for use in L3 update (4)
    t2.map({1,-3}, {-2,4}); //for use ni Q2 update (5)
    t2.map({1}, {3,4,-2});  //for use in Q3 update (6)
    t2.init_amplitudes();
    t2.divide_energy();

    t2temp.init_amplitudes();
    t2temp.divide_energy();
    t2temp.zeros();
    for(uint i = 0; i<t2temp.blocklengths(5); ++i){
        t2temp.getblock(5,i).print();
        cout << endl;
    }
    */




    /*

    uint i = 2;
    t2temp.getblock(0,i).print();
    cout << endl;
    t2temp.getblock(1,i).print();
    cout << endl;
    t2temp.getblock(2,i).print();
    cout << endl;
    t2temp.getblock(3,i).print();
    cout << endl;
    t2temp.getblock(4,i).print();
    cout << endl;
    t2temp.getblock(5,i).print();
    cout << endl;
    t2temp.getblock(6,i).print();
    */


    //t2.vElements.print();

    //field<field<uvec> > test;

    //field<uvec> ab = t2.unpack(AB, R);
    //ab.print();
    //ab.print();


    //ccsolve solver2(fgas);
    //solver2.CCSD_SG(2);
    //ccd solver(fgas, .5);

    //fgas.print();
    //cout << pow(2, 2.0/3.0) << endl;



    /*
    double val = 0;
    int count = 0;
    int Ns = fgas.iNbstates;
    int Nh = fgas.iNparticles;
    for(int a = Nh; a < Ns; ++a){
        for(int i = 0; i < Nh; ++i){
            for(int j = 0; j < Nh; ++j){
                for(int k = 0; k < Nh; ++k){
                    if(fgas.v2(i,a,j,k) != fgas.v2(j,k,i,a)){
                        //cout << fgas.v2(i,a,b,c) << endl;
                        count += 1;
                    }

                }
            }
        }
    }
    cout << "Number of discrepancies: " << count << endl;
    */





    /*

    double val = 0;
    int Ns = fgas.iNbstates;
    int Nh = fgas.iNparticles;
    for(int a = Nh; a < Ns; ++a){
        for(int b = Nh; b < Ns; ++b){
            for(int i = 0; i < Nh; ++i){
                for(int j = 0; j < Nh; ++j){
                    val += .25*fgas.v2(a,b,i,j)*fgas.v2(i,j,a,b)/(fgas.vEnergy(i) +fgas.vEnergy(j) - fgas.vEnergy(a) - fgas.vEnergy(b));


                }
            }
        }
    }
    cout << val/Nh << endl;
    */


    return 0;

}


/*

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
*/
