#include "bccd.h"
#define ARMA_64BIT_WORD
#include <armadillo>

#include "basis/electrongas.h"
#include "solver/amplitude.h"
#include <time.h>

using namespace std;
using namespace arma;


bccd::bccd(electrongas fgas)
{
    eBs = fgas;
    Np = fgas.iNbstates-fgas.iNparticles;
    Nh = fgas.iNparticles; //conflicting naming here
    init();

    //compare();
    cout << "[BCCD]Energy:" << energy() << endl;
    //solve(10);
}



void bccd::init(){
    // ############################################
    // ## Initializing the needed configurations ##
    // ############################################


    //T2 amplitude

    //amplitude tt2(eBs, 8, {Np, Np, Nh, Nh});
    t2.init(eBs, 8, {Np, Np, Nh, Nh});
    //t2 = tt2;
    t2.map({1,2}, {3,4});
    t2.map({2,-4},{-1,3}); //for use in L3 (1)

    t2.map({1,-3},{4,-2}); //for use in Q2 (2)
    t2.map({-4,2},{-1,3}); //for use in Q2 (3)

    t2.map({1,2,-4},{3});  //for use in Q3 (4)
    t2.map({-4,2,1},{3});  //for use in Q3 (5)

    t2.map({1},{4,3,-2});  //for use in Q4 (6)
    t2.map({1},{-2,4,3});  //for use in Q4 (7)



    //t3.map({},{});
    //t2.map({-4,2}, {-1,3}); //for use in L3 update (2)

    t2.init_amplitudes();
    t2.divide_energy();

    //amplitude tt3(eBs, 8, {Np, Np, Nh, Nh});

    //next amplitude
    //t2n = t2;
    t2n.init(eBs, 8, {Np, Np, Nh, Nh});
    t2n.map({1,2}, {3,4});
    t2n.map({2,-4},{-1,3}); //for use in L3 (1)

    t2n.map({1,-3},{4,-2}); //for use in Q2 (2)
    t2n.map({-4,2},{-1,3}); //for use in Q2 (3)

    t2n.map({1,2,-4},{3});  //for use in Q3 (4)
    t2n.map({-4,2,1},{3});  //for use in Q3 (5)

    t2n.map({1},{4,3,-2});  //for use in Q4 (6)
    t2n.map({1},{-2,4,3});  //for use in Q4 (7)


    t2n.init_amplitudes();
    t2n.zeros();

    //Temporary amplitude storage for permutations
    //t2temp = tt3;
    t2temp.init(eBs, 8, {Np, Np, Nh, Nh});
    t2temp.map({1,2},{3,4}); //ab ij (0)
    t2temp.map({2,1},{3,4}); //ba ij (1)
    t2temp.map({1,2},{4,3}); //ab ji (2)
    t2temp.map({2,1},{4,3}); //ba ji (3)

    //the input configurations
    t2temp.map({-4,2}, {-1,3}); //for use in L3 update (4)
    t2temp.map({1,-3}, {-2,4}); //for use ni Q2 update (5)
    t2temp.map({1,2,-4}, {3});  //for use in Q3 update (6)
    t2temp.map({1}, {-2,4,3});  //for use in Q4 update (7)



    t2temp.init_amplitudes();
    t2temp.zeros();

    //t2n.map({1,2}, {3,4});
    //t2n.init_amplitudes();


    //Vhhpp
    //blockmap tvhhpp(eBs, 4, {Nh,Nh,Np,Np});
    //vhhpp = tvhhpp;

    vhhpp.init(eBs, 4, {Nh,Nh,Np,Np});
    vhhpp.init_interaction({0,0,Nh,Nh});
    vhhpp.map({1,-3},{-2,4}); //for use in Q2 (1)
    vhhpp.map({2},{-1,3,4});  //for use in Q3 (2)
    vhhpp.map({1,2,-4},{3});  //for use in Q4 (3)



    //blockmap vv(eBs, 3, {Nh,Nh,Np,Np});
    //v0 = vv;
    v0.init(eBs, 3, {Nh,Nh,Np,Np});
    v0.init_interaction({0,0,Nh,Nh});

    //Vpphh
    //blockmap tvpphh(eBs, 3, {Np, Np, Nh, Nh});
    //vpphh = tvpphh;
    vpphh.init(eBs, 3, {Np, Np, Nh, Nh});
    vpphh.init_interaction({Nh,Nh,0,0});

    //Vpppp
    //blockmap tvpppp(eBs, 3, {Np, Np, Np, Np});
    //vpppp = tvpppp;
    vpppp.init(eBs, 3, {Np, Np, Np, Np});
    vpppp.init_interaction({Nh,Nh,Nh,Nh});

    //Vhhhh
    //blockmap tvhhhh(eBs, 3, {Nh, Nh, Nh, Nh});
    //vhhhh = tvhhhh;
    vhhhh.init(eBs, 3, {Nh, Nh, Nh, Nh});
    vhhhh.init_interaction({0,0,0,0});

    //Vhpph
    //blockmap tvhpph(eBs, 3, {Nh, Np, Np, Nh});
    //vhpph = tvhpph;
    vhpph.init(eBs, 3, {Nh, Np, Np, Nh});
    //vhpph.init_interaction({0,Nh,Nh,0});
    vhpph.map({-4,2},{3,-1}); //for use in L3 (0)


}

void bccd::solve(uint Nt){
    // ############################################
    // ## Solve Nt steps                         ##
    // ############################################

    //set up needed vectors of corresponding blocks in contractions (intersecting configurations)
    umat vpppp_t2 = intersect_blocks(t2,0,vpppp,0);
    umat vpphh_t2 = intersect_blocks(t2,0,vpphh,0);
    umat vhhhh_t2 = intersect_blocks(t2,0,vhhhh,0);
    umat vhpph_L3 = intersect_blocks(t2,1,vhpph,0);

    //quadratic terms
    umat Q1config = intersect_blocks_triple(t2,0,vhhpp,0,t2,0); //this is actually not neccessary to map out, they already align nicely by construction
    umat Q2config = intersect_blocks_triple(t2,2,vhhpp,1,t2,3);
    umat Q3config = intersect_blocks_triple(t2,4,vhhpp,2,t2,5);
    umat Q4config = intersect_blocks_triple(t2,6,vhhpp,3,t2,7);

    //Q1config.print();

    t2n.zeros(); //zero out next amplitudes

    for(uint t = 0; t < Nt; ++t){
        // ############################################
        // ## Reset next amplitude                   ##
        // ############################################

        t2n.zeros();
        for(uint i = 0; i < vpphh_t2.n_rows; ++i){
            mat block = vpphh.getblock(0,vpphh_t2(i,1));
            t2n.addblock(0,vpphh_t2(i,0), block);
        }

        // ############################################
        // ## Calculate L1                           ##
        // ############################################
        for(uint i = 0; i < vpppp_t2.n_rows; ++i){
            mat block = .5*vpppp.getblock(0,vpppp_t2(i,1))*t2.getblock(0,vpppp_t2(i,0));
            t2n.addblock(0,vpppp_t2(i,0), block);
        }

        // ############################################
        // ## Calculate L2                           ##
        // ############################################
        for(uint i = 0; i < vhhhh_t2.n_rows; ++i){
            mat block = .5*t2.getblock(0,vhhhh_t2(i,0))*vhhhh.getblock(0,vhhhh_t2(i,1));
            t2n.addblock(0,vhhhh_t2(i,0), block);
        }

        // ############################################
        // ## Calculate L3                           ##
        // ############################################
        // vhpph*t2 ---
        //fmL3.update(vhpph.sq_rp()*T.qs_pr(), Ns, Nq, Np, Nr);
        //L3 = fmL3.rq_sp() - fmL3.qr_sp() -fmL3.rq_ps() +fmL3.qr_ps();
        t2temp.zeros();
        for(uint i = 0; i < vhpph_L3.n_rows; ++i){
            mat block = vhpph.getblock(0,vhpph_L3(i,1))*t2.getblock(1,vhpph_L3(i,0));
            t2temp.addblock(4,i,block);
        }
        //permute L3
        for(uint i = 0; i < t2temp.fvConfigs(0).n_rows; ++i){
            mat block = t2temp.getblock(0,i) - t2temp.getblock(1,i)- t2temp.getblock(2,i)+t2temp.getblock(3,i);
            t2n.addblock(0,i,block);
        }

        // ############################################
        // ## Calculate Q1                           ##
        // ############################################
        for(uint i = 0; i < Q1config.n_rows; ++i){
            mat block = .25*t2.getblock(0,Q1config(i,0))*(vhhpp.getblock(0,Q1config(i,1))*t2.getblock(0,Q1config(i,2)));
            t2n.addblock(0,Q1config(i,0),block);
        }


        // ############################################
        // ## Calculate Q2                           ##
        // ############################################
        t2temp.zeros();
        for(uint i = 0; i < Q2config.n_rows; ++i){
            mat block = t2.getblock(2,Q2config(i,0))*(vhhpp.getblock(1,Q2config(i,1))*t2.getblock(3,Q2config(i,2)));
            t2temp.addblock(5,Q2config(i,0),block);
        }
        for(uint i = 0; i < t2temp.fvConfigs(0).n_rows; ++i){
            mat block = t2temp.getblock(0,i) - t2temp.getblock(2,i);
            t2n.addblock(0,i,block);
        }

        // ############################################
        // ## Calculate Q3                           ##
        // ############################################
        //fmQ3.update_as_r_pqs((T.r_sqp()*vhhpp.prs_q())*T.r_pqs(), Np, Nq, Nr, Ns); //needs realignment and permutations
        //Q3 = fmQ3.pq_rs() - fmQ3.pq_sr(); //permuting elements
        t2temp.zeros();
        for(uint i = 0; i < Q3config.n_rows; ++i){
            mat block = t2.getblock(4,Q3config(i,0))*(vhhpp.getblock(2,Q3config(i,1))*t2.getblock(5,Q3config(i,2)));
            t2temp.addblock(6,Q3config(i,0),block);
        }
        for(uint i = 0; i < t2temp.fvConfigs(0).n_rows; ++i){
            mat block = -.5*(t2temp.getblock(0,i) - t2temp.getblock(2,i)); //* done to keep in line with equations (inefficient)
            t2n.addblock(0,i,block);
        }

        // ############################################
        // ## Calculate Q4                           ##
        // ############################################
        //fmQ4.update_as_p_qrs(T.p_srq()*vhhpp.pqr_s()*T.p_qrs(), Np, Nq, Nr, Ns); //needs realignment and permutations
        //Q4 = fmQ4.pq_rs() - fmQ4.qp_rs(); //permuting elements
        t2temp.zeros();
        for(uint i = 0; i < Q4config.n_rows; ++i){
            mat block = t2.getblock(6,Q4config(i,0))*(vhhpp.getblock(3,Q4config(i,1))*t2.getblock(7,Q4config(i,2)));
            t2temp.addblock(7,Q4config(i,2),block);
        }
        for(uint i = 0; i < t2temp.fvConfigs(0).n_rows; ++i){
            mat block = .5*(t2temp.getblock(0,i) - t2temp.getblock(1,i)); //* done to keep in line with equations (inefficient)
            t2n.addblock(0,i,block);
        }




        t2n.divide_energy();
        t2 = t2n;
        cout << "[BCCD][" << t << "]Energy:" << energy() << endl;
    }


    //for(uint i = 0; i < vpphh_t2.n_rows; ++i){
    //    t2n.addblock(0,vpppp_t2(i,1), vpphh.getblock(0,vpphh_t2(i,0)));
    //}





}


umat bccd::intersect_blocks_triple(amplitude a, uint na, blockmap b, uint nb, amplitude c, uint nc){
    // #############################################
    // ## Find corresponding blocks in a, b and c ##
    // #############################################
    umat tintersection(a.blocklengths(na), 3);
    uint counter = 0;
    for(uint n1 = 0; n1 < a.blocklengths(na); ++n1){

        int ac = a.fvConfigs(na)(n1);


        for(uint n2 = 0; n2 < b.blocklengths(nb); ++n2){
            int bc = b.fvConfigs(nb)(n2);

            if(bc == ac){
                //found intersecting configuration
                for(uint n3 = 0; n3 < c.blocklengths(nc); ++n3){
                    if(c.fvConfigs(nc)(n3) == ac){
                        //found intersecting configuration
                        tintersection(counter, 0) = n1;
                        tintersection(counter, 1) = n2;
                        tintersection(counter, 2) = n3;
                        counter +=1;
                    }
                }
            }
        }
    }
    //Flatten intersection
    //umat intersection(counter-1, 3);
    umat intersection(counter, 3);
    for(uint n = 0; n < counter; ++n){
        intersection(n, 0) = tintersection(n,0);
        intersection(n, 1) = tintersection(n,1);
        intersection(n, 2) = tintersection(n,2);

    }
    return intersection;
}

umat bccd::intersect_blocks(amplitude a, uint na, blockmap b, uint nb){
    // ############################################
    // ## Find corresponding blocks in a and b   ##
    // ############################################
    umat tintersection(a.blocklengths(na), 2);
    uint counter = 0;
    for(uint n1 = 0; n1 < a.blocklengths(na); ++n1){
        int ac = a.fvConfigs(na)(n1);
        for(uint n2 = 0; n2 < b.blocklengths(nb); ++n2){
            if(b.fvConfigs(nb)(n2) == ac){
                //found intersecting configuration
                tintersection(counter, 0) = n1;
                tintersection(counter, 1) = n2;
                counter +=1;
            }
        }
    }
    //Flatten intersection
    //cout << "Counter:" << counter << endl;
    //umat intersection(counter-1, 2);
    umat intersection(counter, 2); //changed this to avoid valgrind:invalid write of size 8
    for(uint n = 0; n < counter; ++n){
        intersection(n, 0) = tintersection(n,0);
        intersection(n, 1) = tintersection(n,1);
    }
    return intersection;
}

void bccd::compare(){
    for(uint i = 0; i < t2.blocklengths(0); ++i){
        t2.getblock(0,i).print();
        cout << endl;
        vpphh.getblock(0,i).print();
        cout << endl;
        cout << endl;
    }
}

double bccd::energy(){
    //uint n = t2.blocklengths(0);
    double e = 0;
    umat c = intersect_blocks(t2,0,vhhpp,0); //this should be calculated prior to function calls (efficiency)
    for(uint i = 0; i < c.n_rows; ++i){
        mat block = vhhpp.getblock(0, c(i,1))*t2.getblock(0,c(i,0));
        vec en = block.diag();
        for(uint j = 0; j< en.n_rows; ++j){
            e += en(j);
        }
    }
    return .25*e;
}
