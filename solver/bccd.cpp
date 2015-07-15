#include "bccd.h"
#define ARMA_64BIT_WORD
#include <armadillo>

#include "basis/electrongas.h"
#include "solver/amplitude.h"
#include <time.h>
#include <omp.h>

using namespace std;
using namespace arma;


bccd::bccd(electrongas fgas)
{
    eBs = fgas;
    Np = fgas.iNbstates-fgas.iNparticles;
    Nh = fgas.iNparticles; //conflicting naming here
    init();
    cout << "[BCCD]Energy:" << energy() << endl;
    solve(10);
}



void bccd::init(){
    // ############################################
    // ## Initializing the needed configurations ##
    // ############################################

    pert_triples = true;

    clock_t t;
    t = clock();


    t2.init(eBs, 10, {Np, Np, Nh, Nh});
    t2.map_t2_permutations();
    t2.map({2,-4},{-1,3}); //for use in L3 (1)

    t2.map({1,-3},{4,-2}); //for use in Q2 (2)
    t2.map({-4,2},{-1,3}); //for use in Q2 (3)

    t2.map({1,2,-4},{3});  //for use in Q3 (4)
    t2.map({-4,2,1},{3});  //for use in Q3 (5)

    t2.map({1},{4,3,-2});  //for use in Q4 (6)
    t2.map({1},{-2,4,3});  //for use in Q4 (7)

    if(pert_triples){
        t2.map({2},{-1,2,3});  //for use in t2a (8)
        t2.map({1,2,-3},{4});  //for use in t2b (9)
    }


    t2.init_amplitudes();
    t2.divide_energy();
    vec nz = unique(t2.vElements);

    //next amplitude
    t2n = t2;

    t2n.zeros();

    //Temporary amplitude storage for permutations
    t2temp2.init(eBs, 8, {Np, Np, Nh, Nh});
    t2temp2.map_t2_permutations();
    t2temp2.init_amplitudes();
    t2temp2.map({-4,2}, {-1,3}); //for use in L3 update (1)
    t2temp2.map({1,-3}, {-2,4}); //for use ni Q2 update (2)
    t2temp2.map({1,2,-4}, {3});  //for use in Q3 update (3)
    t2temp2.map({1}, {-2,4,3});  //for use in Q4 update (4)


    if(pert_triples){
        t2temp2.map({2}, {3,4,-1});  //for use in D10b update (5)
        t2temp2.map({1,2,-3}, {4});  //for use in D10c update (6)

    }
    t2temp.init_amplitudes();
    t2temp.zeros();

    vhhpp.init(eBs, 4, {Nh,Nh,Np,Np});
    vhhpp.init_interaction({0,0,Nh,Nh});
    vhhpp.map({1,-3},{-2,4}); //for use in Q2 (1)
    vhhpp.map({2},{-1,3,4});  //for use in Q3 (2)
    vhhpp.map({1,2,-4},{3});  //for use in Q4 (3)

    v0.init(eBs, 3, {Nh,Nh,Np,Np});
    v0.init_interaction({0,0,Nh,Nh});

    vpphh.init(eBs, 3, {Np, Np, Nh, Nh});
    vpphh.init_interaction({Nh,Nh,0,0});


    vpppp.init(eBs, 3, {Np, Np, Np, Np});
    vpppp.map_vpppp();

    vhhhh.init(eBs, 3, {Nh, Nh, Nh, Nh});
    vhhhh.init_interaction({0,0,0,0});

    vhpph.init(eBs, 3, {Nh, Np, Np, Nh});
    vhpph.map({-4,2},{3,-1}); //for use in L3 (0)
    cout << "[BCCD] Time spent initializing doubles.:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
    t = clock();

    // #############################################
    // ## Perturbative triples setup              ##
    // #############################################
    if(pert_triples){
        vhphh.init(eBs, 3, {Nh, Np, Nh, Nh});
        vhphh.map({1},{-2,3,4});

        vppph.init(eBs, 3, {Np, Np, Np, Nh});
        vppph.map({1,2,-3},{4});

        vphpp.init(eBs, 3, {Np, Nh, Np, Np});
        vphpp.map({1},{-2,3,4});

        vhhhp.init(eBs, 3, {Nh,Nh,Nh,Np});
        vhhhp.map({1,2,-4},{3});
        cout << "[BCCD] Time spent initializing triples interaction:" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
        t = clock();


        t3.init(eBs, 3, {Np, Np, Np, Nh, Nh, Nh});
        t3.make_t3();
        t3.map_t3_permutations();
        t3temp = t3;

        t3.map_t3_623_451(vphpp.fvConfigs(0)); //for d10b
        t3.map_t3_124_356(vhhhp.fvConfigs(0)); //for d10c

        //t3.map6({-6,2,3}, {4,5,-1}); //for use in d10b (1) //these need special treatment
        //t3.map6({1,2,-4}, {-3,5,6}); //for use in d10c (2) (and

        t3.init_t3_amplitudes();

        cout << "[BCCD] Time spent mapping T3 amplitude (1):" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
        t = clock();

        t3temp.map_t3_236_145(t2.fvConfigs(8));
        t3temp.map6({1,2,-4}, {-3,5,6}); //, t2.fvConfigs(6)); //for use in t2b (2)
        t3temp.init_t3_amplitudes(); //HUGE opt-potential (use precalc F)
        t3temp.zeros();
        cout << "[BCCD] Time spent mapping T3 amplitude (2):" << (float)(clock()-t)/CLOCKS_PER_SEC << endl;
        t = clock();    }

    //print t3temp configurations

    /*
    for(uint i = 0; i < t3temp.blocklengths(1); ++i){
        t3temp.getblock(1,i).print();
        cout << t3temp.getblock(1, i).max() << " " << i << endl;
        cout << endl;
    }*/


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

    umat t2aconfig = intersect_blocks_triple(t2,8, vppph, 0, t3temp, 1);
    //umat t2bconfig = intersect_blocks(t2,9, vhphh, 0);
    umat t2bconfig = intersect_blocks_triple(t2,9, vhphh, 0, t3temp, 2);
    //t3temp.fvConfigs(2).print();
    //t2bconfig.print();

    umat d10bconfig = intersect_blocks_triple(t3,1, vphpp, 0, t2temp2, 5);
    umat d10cconfig = intersect_blocks_triple(t3,2, vhhhp, 0, t2temp2, 6);

    t2n.zeros(); //zero out next amplitudes

    uint nthreads = 4;
    for(uint t = 0; t < Nt; ++t){
        // ############################################
        // ## Reset next amplitude                   ##
        // ############################################
        t2n.zeros();
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < vpphh_t2.n_rows; ++i){
            mat block = vpphh.getblock(0,vpphh_t2(i,1));
            t2n.addblock(0,vpphh_t2(i,0), block);
        }

        // ############################################
        // ## Calculate L1                           ##
        // ############################################
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < vpppp_t2.n_rows; ++i){
            mat block = .5*vpppp.getblock_vpppp(0,vpppp_t2(i,1))*t2.getblock(0,vpppp_t2(i,0));
            t2n.addblock(0,vpppp_t2(i,0), block);
        }


        // ############################################
        // ## Calculate L2                           ##
        // ############################################
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < vhhhh_t2.n_rows; ++i){
            mat block = .5*t2.getblock(0,vhhhh_t2(i,0))*vhhhh.getblock(0,vhhhh_t2(i,1));
            t2n.addblock(0,vhhhh_t2(i,0), block);
        }

        // ############################################
        // ## Calculate L3                           ##
        // ############################################
        t2temp2.zeros();
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < vhpph_L3.n_rows; ++i){
            mat block = vhpph.getblock(0,vhpph_L3(i,1))*t2.getblock(1,vhpph_L3(i,0));
            t2temp2.addblock(1,i,block);
        }
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < t2temp2.fvConfigs(0).n_rows; ++i){
            mat block2 = t2temp2.getblock(0,i) - t2temp2.getblock_permuted(0,i,3) - t2temp2.getblock_permuted(0,i,0) + t2temp2.getblock_permuted(0,i,6);
            t2n.addblock(0,i,block2);
        }

        // ############################################
        // ## Calculate Q1                           ##
        // ############################################
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < Q1config.n_rows; ++i){
            mat block = .25*t2.getblock(0,Q1config(i,0))*(vhhpp.getblock(0,Q1config(i,1))*t2.getblock(0,Q1config(i,2)));
            t2n.addblock(0,Q1config(i,0),block);
        }


        // ############################################
        // ## Calculate Q2                           ##
        // ############################################
        t2temp2.zeros();
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < Q2config.n_rows; ++i){
            mat block = t2.getblock(2,Q2config(i,0))*(vhhpp.getblock(1,Q2config(i,1))*t2.getblock(3,Q2config(i,2)));
            t2temp2.addblock(2,Q2config(i,0),block);
        }
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < t2temp2.fvConfigs(0).n_rows; ++i){
            //mat block = t2temp.getblock(0,i) - t2temp.getblock(2,i);
            mat block2 = t2temp2.getblock(0,i) - t2temp2.getblock_permuted(0,i,0);
            t2n.addblock(0,i,block2);
        }

        // ############################################
        // ## Calculate Q3                           ##
        // ############################################
        t2temp2.zeros();
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < Q3config.n_rows; ++i){
            mat block = t2.getblock(4,Q3config(i,0))*(vhhpp.getblock(2,Q3config(i,1))*t2.getblock(5,Q3config(i,2)));
            t2temp2.addblock(3,Q3config(i,0),block);
        }
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < t2temp2.fvConfigs(0).n_rows; ++i){
            mat block2 = -.5*(t2temp2.getblock(0,i) - t2temp2.getblock_permuted(0,i,3));
            t2n.addblock(0,i,block2);
        }

        // ############################################
        // ## Calculate Q4                           ##
        // ############################################
        t2temp2.zeros();
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < Q4config.n_rows; ++i){
            mat block = t2.getblock(6,Q4config(i,0))*(vhhpp.getblock(3,Q4config(i,1))*t2.getblock(7,Q4config(i,2)));
            t2temp2.addblock(4,Q4config(i,2),block);
        }
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < t2temp2.fvConfigs(0).n_rows; ++i){
            mat block2 = .5*(t2temp2.getblock(0,i) - t2temp2.getblock_permuted(0,i,0));
            t2n.addblock(0,i,block2);
        }

        if(pert_triples){
            // ##################################################
            // ##                                              ##
            // ## Calculate perturbative triples amplitudes    ##
            // ##                                              ##
            // ##################################################
            t3temp.zeros();

            // ############################################
            // ## Calculate t2a                           ##
            // ############################################

            for(uint i = 0; i < t2aconfig.n_rows; ++i){
                mat block = vppph.getblock(0,t2aconfig(i,1))*t2.getblock(8,t2aconfig(i,0));
                //t2.getblock(6,t2aconfig(i,0))*vhhpp.getblock(3,Q4config(i,1))*t2.getblock(7,Q4config(i,2)));
                t3temp.addblock(1,t2aconfig(i,2),block); //t2aconfig is wriong
            }
            for(uint i = 0; i < t3temp.fvConfigs(0).n_rows; ++i){
                //cout << i << endl;
                mat block = t3temp.getblock(0,i)
                                - t3temp.getblock_permuted(0,i,0)
                                - t3temp.getblock_permuted(0,i,1)
                                - t3temp.getblock_permuted(0,i,4)
                                + t3temp.getblock_permuted(0,i,7)
                                + t3temp.getblock_permuted(0,i,10)
                                - t3temp.getblock_permuted(0,i,5)
                                + t3temp.getblock_permuted(0,i,8)
                                + t3temp.getblock_permuted(0,i,11)
                                ;
                t3.addblock(0,i,block);
            }


            // ############################################
            // ## Calculate t2b                           ##
            // ############################################
            t3temp.zeros();
            //double tsum = 0;
            //double vsum = 0;
            //double bsum = 0;
            for(uint i = 0; i < t2bconfig.n_rows; ++i){
                //sum all these and compare
                //vsum += sum(abs(vectorise(vhphh.getblock(0,t2bconfig(i,1))))); // << endl;
                //tsum += sum(abs(vectorise(t2.getblock(9,t2bconfig(i,0)))));
                mat block = t2.getblock(9,t2bconfig(i,0))*vhphh.getblock(0,t2bconfig(i,1)); //.t();
                //bsum += sum(abs(vectorise(block)));

                t3temp.addblock(2,t2bconfig(i,2),block); //t2bconfig is wrong
            }
            //cout << vsum << " " << tsum << " ";
            //cout << bsum << endl;
            //bsum = 0;
            for(uint i = 0; i < t3temp.fvConfigs(0).n_rows; ++i){
                //1 - ac - cb - ij + ij/ac + ij/cb - ik +ik/ac + ik/cb
                mat block = -1.0*(t3temp.getblock(0,i)
                                - t3temp.getblock_permuted(0,i,1)
                                - t3temp.getblock_permuted(0,i,2)
                                - t3temp.getblock_permuted(0,i,3)
                                + t3temp.getblock_permuted(0,i,9)
                                + t3temp.getblock_permuted(0,i,12)
                                - t3temp.getblock_permuted(0,i,4)
                                + t3temp.getblock_permuted(0,i,10)
                                + t3temp.getblock_permuted(0,i,13)
                                );
                //bsum += sum(abs(vectorise(block)));
                t3.addblock(0,i,block);
            }
            //cout << bsum << endl;

            // ############################################
            // ## Set up T3                           ##
            // ############################################

            t3.divide_energy();


            // ############################################
            // ## Calculate D10b                           ##
            // ############################################

            t2temp2.zeros();
            //#pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < d10bconfig.n_rows; ++i){
                mat block = vphpp.getblock(0,d10bconfig(i,1))*t3.getblock(1, d10bconfig(i,0));
                t2temp2.addblock(5,d10bconfig(i,2),block);
            }
            //#pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < t2temp2.fvConfigs(0).n_rows; ++i){
                //mat block = t2temp.getblock(0,i) - t2temp.getblock(2,i);
                mat block2 = -.5*(t2temp2.getblock(0,i) - t2temp2.getblock_permuted(0,i,0));
                t2n.addblock(0,i,block2);
            }

            t2temp2.zeros();

            // ############################################
            // ## Calculate D10c                           ##
            // ############################################

            //#pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < d10cconfig.n_rows; ++i){
                mat block = t3.getblock(2, d10cconfig(i,0))*vhhhp.getblock(0,d10cconfig(i,1));
                t2temp2.addblock(6,d10cconfig(i,2),block);
            }
            //#pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < t2temp2.fvConfigs(0).n_rows; ++i){
                //mat block = t2temp.getblock(0,i) - t2temp.getblock(2,i);
                mat block2 = .5*(t2temp2.getblock(0,i) - t2temp2.getblock_permuted(0,i,3));
                t2n.addblock(0,i,0*block2);
            }


        }

        t2n.divide_energy();
        t2 = t2n;
        cout << "[BCCD][" << t << "]Energy:" << energy() << endl;
    }
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
