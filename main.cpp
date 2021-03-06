#define ARMA_64BIT_WORD
#include <armadillo>

#include <iomanip>
#include <fstream>
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
#include "solver/sccdt_mp.h"
#include <omp.h>

using namespace std;
using namespace arma;


int main(int argc, char *argv[] )
{
    //default setup
    int i0 = 3;
    int i1 = 4;
    uint iNp = 14;
    double rs = 1.0;
    uint iterations = 20;
    uint uiStatAlloc = 10000000;
    uint uiMode = 2;
    double dConvergenceThreshold = 0.0000000001;
    double dRelaxation = .3;

    if ( argc == 10 ){
        i0 = atoi(argv[1]);
        i1 = atoi(argv[2]);

        iNp = atoi(argv[3]);

        rs = atof(argv[4]);

        iterations = atoi(argv[5]);
        dConvergenceThreshold = pow(10.0,-1*atof(argv[6]));
        dRelaxation = atof(argv[7]);

        uiStatAlloc = atoi(argv[8]);
        uiMode = atoi(argv[9]);
        cout << "start/end        :" << i0 << "/" << i1 << endl;
        cout << "Np               :" << iNp<< endl;
        cout << "rs               :" << rs << endl;
        cout << "iterations       :" << iterations << endl;
        cout << "precision        :" << dConvergenceThreshold << endl;
        cout << "relaxation       :" << dRelaxation << endl;
        cout << "static allocation:" << uiStatAlloc << endl;


    }
    else{
        cout << "FermiCC, by Audun Skau Hansen 2015." << endl;
        cout << "Usage:" << endl;
        cout << "./FermiCC [start] [end] [Np] [rs] [iterations] [precision] [relaxation] [static allocation] [mode]" << endl;
        cout << endl;
        cout << "start/end        :       Number of the first shell (start) and up to (but not including) the last shell (end)."<< endl;
        cout << "Np               :       Number of electrons." << endl;
        cout << "rs               :       Wigner-Seiz radius. See thesis." << endl;
        cout << "iterations       :       Maximum number of iterations." << endl;
        cout << "precision        :       Number of decimals in answer" << endl;
        cout << "relaxation       :       Relaxation for amplitude updates." << endl;
        cout << "static allocation:       Maximum size of statically allocated vectors. (use 100000 on smaller calculations.)" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "modes            :       Block: 1 CCD" << endl;
        cout << "                                2 CCDT-1" << endl;
        cout << "                         Sparse:3 CCD" << endl;
        cout << "                                4 CCDT-1" << endl;
        cout << "                                5 CCDT" << endl;
        cout << "---------------------------------------------------------------" << endl;
        cout << "Running default calculation, i.e.:" << endl;
        cout << "./FermiCC 3 4 14 1.0 20 10 100000 1" << endl;

    }


    //electrongas fgas;
    //fgas.generate_state_list2(4.0,2.0, 14);
    //cout << "[Main] " << setprecision(16) << "Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    //cout << "[Main] G.Baardsens results:" << 1.9434 << endl;


    //uint Np = fgas.iNbstates-fgas.iNparticles; //conflicting notation here
    //uint Nh = fgas.iNparticles;


    //bccd solver1(fgas,.3);





    for(int i = i0; i < i1; ++i){
        electrongas fgas;
        fgas.generate_state_list2(i,rs, iNp);
        cout << setprecision(10) << rs << "  " << fgas.iNbstates << endl;
        //sccdt_mp solve(fgas, .3);
        if(uiMode==1){
            bccd solver1(fgas,.3, dConvergenceThreshold);
            solver1.pert_triples = false;
            solver1.uiStatAlloc = uiStatAlloc;
            solver1.init();
            solver1.solve(iterations);
            cout << "Number of states, correlation energy, difference in last convergence, number of iterations passed" << endl;

            cout << setprecision(9) << "[CCD (block)]" << fgas.iNbstates << "     " << solver1.dCorrelationEnergy << "     " << solver1.convergence_diff << "     "<< solver1.convergence << endl;
        }
        if(uiMode==2){
            bccd solver1(fgas,.3, dConvergenceThreshold);
            solver1.uiStatAlloc = uiStatAlloc;
            solver1.init();
            solver1.solve(iterations);
            cout << "Number of states, correlation energy, difference in last convergence, number of iterations passed" << endl;

            cout << setprecision(9) << "[CCDT-1 (block)]"<< fgas.iNbstates << "     " << solver1.dCorrelationEnergy << "     " << solver1.convergence_diff << "     "<< solver1.convergence << endl;
        }

        if(uiMode==3){
            ccd_mp solver1(fgas,.3);
            //solver1.uiStatAlloc = uiStatAlloc;
            //solver1.init();
            solver1.solve(iterations, dRelaxation, dConvergenceThreshold);
            cout << "Number of states, correlation energy, difference in last convergence, number of iterations passed" << endl;

            cout << setprecision(9) << "[CCD (sparse)]"<< fgas.iNbstates << "     " << solver1.correlation_energy <<  endl;
        }

        if(uiMode==4){
            cout << "Mode not available in this version." << endl;

            //ccd_mp solver1(fgas,.3);
            //solver1.uiStatAlloc = uiStatAlloc;
            //solver1.init();
            //solver1.solve(iterations, dRelaxation, dConvergenceThreshold);
            //cout << "Number of states, correlation energy, difference in last convergence, number of iterations passed" << endl;

            //cout << setprecision(9) << "[CCD (sparse)]"<< fgas.iNbstates << "     " << solver1.correlation_energy <<  endl;
        }

        if(uiMode==5){
            sccdt_mp solver1(fgas,dRelaxation);
            //solver1.uiStatAlloc = uiStatAlloc;
            //solver1.init();
            solver1.solve(iterations, dConvergenceThreshold);
            cout << "Number of states, correlation energy, difference in last convergence, number of iterations passed" << endl;

            cout << setprecision(9) << "[CCD (sparse)]"<< fgas.iNbstates << "     " << solver1.correlation_energy <<  endl;
        }


    }



    /*
    electrongas eBs;
    eBs.generate_state_list2(29,1.0, 14);
    u64 Np = eBs.iNbstates-eBs.iNparticles;
    u64 Nh = 14;

    u64 N = Np*(Np+1)*(Np+2)/6;
    //indices
    ivec a(N);
    ivec b(N);
    ivec c(N);
    u64 count = 0;
    for(int na = 0; na<Np; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            for(int nc = 0; nc<nb+1; ++nc){
                //a(count) = na;
                //b(count) = nb;
                //c(count) = nc;
                count += 1;
            }
        }
    }
    cout << Np << " " << Nh << " " << count << " " << N << endl;
    //ivec Kabc = eBs.unique(conv_to<uvec>::from(a) +Nh) + eBs.unique(conv_to<uvec>::from(b)+Nh) + eBs.unique(conv_to<uvec>::from(c) +Nh);

    //field<ivec> pppmap(4);
    //pppmap(0) = a;
    //pppmap(1) = b;
    //pppmap(2) = c;
    //pppmap(3) = Kabc;
    */


    return 0;


}
