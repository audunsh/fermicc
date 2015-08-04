#include "bccd.h"
#define ARMA_64BIT_WORD
#include <armadillo>

#include "basis/electrongas.h"
#include "solver/amplitude.h"
#include <time.h>
#include <omp.h>

using namespace std;
using namespace arma;


bccd::bccd(electrongas fgas, double relaxation)
{
    eBs = fgas;
    Np = fgas.iNbstates-fgas.iNparticles;
    Nh = fgas.iNparticles; //conflicting naming here
    activate_diagrams();
    nthreads = 5;
    init();
    cout << setprecision(16) <<"[BCCD]Energy:" << energy() << endl;

    dRelaxation_parameter = relaxation;
    dCorrelationEnergy = 1000;
    dTreshold = 0.0000000000000001;

    t3.nthreads = nthreads;
    solve(36);
}

void bccd::activate_diagrams(){
    // #########################
    // ## Choose the diagrams that contribute
    // #########################

    acL1 = 1;
    acL2 = 1;
    acL3 = 1;
    acL4 = 1;

    acQ1 = 1;
    acQ2 = 1;
    acQ3 = 1;
    acQ4 = 1;

    acT2a = 1;
    acT2b = 1;

    acD10b = 1;
    acD10c = 1;




}



void bccd::init(){
    // ############################################
    // ## Initializing the needed configurations ##
    // ############################################
    mode = "CCD";
    pert_triples = true;

    clock_t t;
    t = clock();

    double tm = omp_get_wtime();


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
        //t2.map({2},{-1,2,3});  //for use in t2a (8)
        t2.map({2},{-1,3,4});  //for use in t2a (8)

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

    cout << "[" << mode << "] Time spent initializing doubles.:" << omp_get_wtime()-tm << endl;
    t = clock();
    tm = omp_get_wtime();

    // #############################################
    // ## Perturbative triples setup              ##
    // #############################################
    if(pert_triples){
        mode = "CCDT-1";
        vhphh.init(eBs, 3, {Nh, Np, Nh, Nh});
        vhphh.map({1},{-2,3,4});

        vppph.init(eBs, 3, {Np, Np, Np, Nh});

        vppph.map({1,2,-4},{3});


        vphpp.init(eBs, 3, {Np, Nh, Np, Np});
        vphpp.map({1},{-2,3,4});

        vhhhp.init(eBs, 3, {Nh,Nh,Nh,Np});
        vhhhp.map({-4, 1,2},{3});
        cout << "[" << mode << "] Time spent initializing triples interaction:" << omp_get_wtime()-tm << endl;
        tm = omp_get_wtime();

        t3.init(eBs, 5, {Np, Np, Np, Nh, Nh, Nh});
        t3.nthreads = nthreads;
        t3.make_t3();
        t3.uiCurrent_block = 1;
        //t3.map_t3_permutations();




        t3.map_t3_623_451(vphpp.fvConfigs(0)); //for d10b,
        t3.map_t3_124_356(vhhhp.fvConfigs(0)); //for d10c + t2.fvConfigs(9)
        t3.map_t3_236_145(t2.fvConfigs(8)); //check this one
        //count memory usage in maps
        /*
        u64 memsize2 = 0;
        for(uint u = 1; u < 4; ++u){
            for(uint i = 0; i < t3.fmBlocks(u).n_rows; ++i){
                memsize2 += t3.fmBlocks(u)(i).n_elem;
            }
        }
        cout << "Memory usage from blocks:" << (memsize2*4)/(1000000000.0) <<  " Gb" << endl;
        */



        t3.map_t3_permutations_bconfig_sparse();

        //cout << "Number of enrolled (sparse) states (t3):" << t3.debug_enroll << " of " << t3.uvElements.n_rows << endl;

        t3.init_t3_amplitudes();
        //cout << "Memory usage from elements:" << (3*t3.vElements.n_elem*8)/(1000000000.0) <<  " Gb" << endl;

        cout << "[" << mode << "] Approx of initialized memory for t3    :" << t3.memsize*8/1000000000.0 << " GB" << endl;

        /*
        t3temp.map_t3_236_145(t2.fvConfigs(8)); //check this one
        t3temp.map_t3_124_356(t2.fvConfigs(9));

        t3temp.ivBconfigs = t3.ivBconfigs;



        t3temp.debug_enroll = 0;
        t3temp.map_t3_permutations_bconfig_sparse();
        cout << "Number of enrolled (sparse) states:" << t3temp.debug_enroll << " of " << t3temp.uvElements.n_rows << endl;

        t3temp.init_t3_amplitudes(); //HUGE opt-potential (use precalc F)
        t3temp.zeros();



        cout << "[" << mode << "] Time spent mapping T3 amplitude (2):" << omp_get_wtime()-tm << endl;

        cout << "[" << mode << "] Number of initialized values for t3    :" << t3.memsize << endl;
        cout << "[" << mode << "] Approx of initialized memory for t3    :" << t3.memsize*8/1000000000.0 << " GB" << endl;

        cout << "[" << mode << "] Number of initialized values for t3temp:" << t3temp.memsize << endl;
        */



    }
    //t3temp.insert_zeros();
    t3.insert_zeros();




}

void bccd::solve(uint Nt){
    // ############################################
    // ## Solve Nt steps                         ##
    // ############################################
    double tm = omp_get_wtime();

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


    //t3temp.fvConfigs(2).print();
    //t2bconfig.print();
    umat t2aconfig, t2bconfig, d10bconfig, d10cconfig;
    if(pert_triples){
        t2aconfig = intersect_blocks_triple(t2,8, vppph, 0, t3, 3);
        t2bconfig = intersect_blocks_triple(t2,9, vhphh, 0, t3, 2);

        d10bconfig = intersect_blocks_triple(t3,1, vphpp, 0, t2temp2, 5);
        d10cconfig = intersect_blocks_triple(t3,2, vhhhp, 0, t2temp2, 6);
    }


    //t2.fvConfigs(8).print(); //should try to map this manually (do a "full" mapping)
    //cout << endl;
    //vppph.fvConfigs(0).print(); //does t2a even contribute?!
    //cout << endl;
    //t3temp.fvConfigs(1).print();


    //t2aconfig.print();


    t2n.zeros(); //zero out next amplitudes
    //cout << "[" << mode << "] Time spent intersecting blocks:" << omp_get_wtime()-tm << endl;
    tm = omp_get_wtime();

    //uint nthreads = 5;
    for(uint t = 0; t < Nt; ++t){

        double tm = omp_get_wtime();
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
            t2n.addblock(0,vpppp_t2(i,0), acL1*block);
        }


        // ############################################
        // ## Calculate L2                           ##
        // ############################################
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < vhhhh_t2.n_rows; ++i){
            mat block = .5*t2.getblock(0,vhhhh_t2(i,0))*vhhhh.getblock(0,vhhhh_t2(i,1));
            t2n.addblock(0,vhhhh_t2(i,0), acL2*block);
        }

        // ############################################
        // ## Calculate L3                           ##
        // ############################################
        t2temp2.zeros();
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < vhpph_L3.n_rows; ++i){
            mat block = vhpph.getblock(0,vhpph_L3(i,1))*t2.getblock(1,vhpph_L3(i,0));
            t2temp2.addblock(1,i,acL3*block);
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
            t2n.addblock(0,Q1config(i,0),acQ1*block);
        }


        // ############################################
        // ## Calculate Q2                           ##
        // ############################################
        t2temp2.zeros();
        #pragma omp parallel for num_threads(nthreads)
        for(uint i = 0; i < Q2config.n_rows; ++i){
            mat block = t2.getblock(2,Q2config(i,0))*(vhhpp.getblock(1,Q2config(i,1))*t2.getblock(3,Q2config(i,2)));
            t2temp2.addblock(2,Q2config(i,0),acQ2*block);
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
            t2temp2.addblock(3,Q3config(i,0),acQ3*block);
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
            t2temp2.addblock(4,Q4config(i,2),acQ4*block);
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
            //t3temp.zeros();
            t3.zeros();
            t3.tempZeros();

            // ############################################
            // ## Calculate t2a                           ##
            // ############################################




            #pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < t2aconfig.n_rows; ++i){
                mat block = vppph.getblock(0,t2aconfig(i,1))*t2.getblock(8,t2aconfig(i,0));

                //t3temp.addblock(1,t2aconfig(i,2),acT2a*block);
                t3.addblock_temp(3,t2aconfig(i,2),acT2a*block);

            }


            //if(t3temp.vElements(0) !=0){
            //    cout << "Value error in vElements" << endl;
            //}

            #pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < t3.fvConfigs(4).n_rows; ++i){
                //for(uint i = 0; i < t3temp.fvConfigs(3).n_rows; ++i){
                uint b = 3;
                /*
                mat block = t3temp.getsblock(3,i)
                                - t3temp.getsblock_permuted(3,i,0)
                                - t3temp.getsblock_permuted(3,i,1)
                                - t3temp.getsblock_permuted(3,i,4)
                                + t3temp.getsblock_permuted(3,i,7)
                                + t3temp.getsblock_permuted(3,i,10)
                                - t3temp.getsblock_permuted(3,i,5)
                                + t3temp.getsblock_permuted(3,i,8)
                                + t3temp.getsblock_permuted(3,i,11)
                                ;*/
                mat block2 = t3.getsblock_temp(3,i)
                                - t3.getsblock_permuted_temp(3,i,0)
                                - t3.getsblock_permuted_temp(3,i,1)
                                - t3.getsblock_permuted_temp(3,i,4)
                                + t3.getsblock_permuted_temp(3,i,7)
                                + t3.getsblock_permuted_temp(3,i,10)
                                - t3.getsblock_permuted_temp(3,i,5)
                                + t3.getsblock_permuted_temp(3,i,8)
                                + t3.getsblock_permuted_temp(3,i,11)
                                ;


                t3.addsblock(0,i,block2);
            }



            // ############################################
            // ## Calculate t2b                           ##
            // ############################################
            t3temp.zeros();
            t3.tempZeros();


            #pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < t2bconfig.n_rows; ++i){
                mat block = t2.getblock(9,t2bconfig(i,0))*vhphh.getblock(0,t2bconfig(i,1)); //.t();
                //t3temp.addblock(2,t2bconfig(i,2),acT2b*block); //t2bconfig is wrong
                t3.addblock_temp(2,t2bconfig(i,2),acT2b*block);
            }

            //if(t3temp.vElements(0) !=0){
            //    cout << "Value error in vElements" << endl;
            //}

            #pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < t3.fvConfigs(4).n_rows; ++i){
                //1 - ac - cb - ij + ij/ac + ij/cb - ik +ik/ac + ik/cb
                //umat aligned = t3temp.getfspBlock(i);
                /*
                mat block = -1.0*(t3temp.getsblock(3,i)
                                - t3temp.getsblock_permuted(3,i,1)
                                - t3temp.getsblock_permuted(3,i,2)
                                - t3temp.getsblock_permuted(3,i,3)
                                + t3temp.getsblock_permuted(3,i,9)
                                + t3temp.getsblock_permuted(3,i,12)
                                - t3temp.getsblock_permuted(3,i,4)
                                + t3temp.getsblock_permuted(3,i,10)
                                + t3temp.getsblock_permuted(3,i,13)
                                );*/

                mat block2 = -1.0*(t3.getsblock_temp(3,i)
                                - t3.getsblock_permuted_temp(3,i,1)
                                - t3.getsblock_permuted_temp(3,i,2)
                                - t3.getsblock_permuted_temp(3,i,3)
                                + t3.getsblock_permuted_temp(3,i,9)
                                + t3.getsblock_permuted_temp(3,i,12)
                                - t3.getsblock_permuted_temp(3,i,4)
                                + t3.getsblock_permuted_temp(3,i,10)
                                + t3.getsblock_permuted_temp(3,i,13)
                                );

                t3.addsblock(0,i,block2);
            }
            // ############################################
            // ## Set up T3                           ##
            // ############################################
            t3.divide_energy();



            // ############################################
            // ## Calculate D10b                           ##
            // ############################################

            t2temp2.zeros();

            #pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < d10bconfig.n_rows; ++i){
                mat block = vphpp.getblock(0,d10bconfig(i,1))*t3.getblock(1, d10bconfig(i,0));
                t2temp2.addblock(5,d10bconfig(i,2),acD10b*block);

            }
            #pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < t2temp2.fvConfigs(0).n_rows; ++i){
                mat block2 = .5*(t2temp2.getblock(0,i) - t2temp2.getblock_permuted(0,i,0));
                t2n.addblock(0,i,block2);
            }

            t2temp2.zeros();

            // ############################################
            // ## Calculate D10c                           ##
            // ############################################

            #pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < d10cconfig.n_rows; ++i){
                mat block = t3.getblock(2, d10cconfig(i,0))*vhhhp.getblock(0,d10cconfig(i,1));
                t2temp2.addblock(6,d10cconfig(i,2),acD10c*block);
            }

            #pragma omp parallel for num_threads(nthreads)
            for(uint i = 0; i < t2temp2.fvConfigs(0).n_rows; ++i){
                mat block2 = -.5*(t2temp2.getblock(0,i) - t2temp2.getblock_permuted(0,i,3));
                t2n.addblock(0,i,block2);
            }


        }

        t2n.divide_energy();
        t2.vElements = t2.vElements*dRelaxation_parameter + t2n.vElements*(1-dRelaxation_parameter); //relaxation
        double dNewEnergy = energy();
        if(abs(dCorrelationEnergy-dNewEnergy)<dTreshold){
            t = Nt;

        }
        dCorrelationEnergy = dNewEnergy;
        //t2 =t2n;
        cout << "[" << mode << "][" << t << "]Energy:" << energy() << endl;
        //cout << "[" << mode << "]   Iteration time:" << omp_get_wtime()-tm << endl;
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
