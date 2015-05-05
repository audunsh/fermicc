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


//#include <eigen/Eigen/Dense>
//#include <eigen/Eigen/Sparse>



using namespace std;
using namespace arma;


int main()
{
    electrongas fgas;
    fgas.generate_state_list(20.0,1.0, 14);

    //cout << "Energy per particle:" << fgas.eref(14)/14.0 << " (a.u)"  << endl;
    cout << "# Energy per particle:" << 2*fgas.eref(14)/14.0 << " (rydberg)"  << endl;
    cout << "# G.Baardsens results:" << 1.9434 << endl;

    ccd solver(fgas);
    //fgas.print();
    //cout << pow(2, 2.0/3.0) << endl;

    /*
    double val = 0;
    for(int p = 0; p < fgas.iNbstates; ++p){
        for(int q = 0; q < fgas.iNbstates; ++q){
            for(int r = 0; r < fgas.iNbstates; ++r){
                for(int s = 0; s < fgas.iNbstates; ++s){
                    val = fgas.v2(p,q,r,s);
                    if(val != 0){
                        cout << p << " "<< q << " "<< r << " "<< s  << " " << val << endl;
                    }
                }
            }
        }
    }
    */








    //testing sp_mat initialization time
    /*
    clock_t t0;
    sp_mat M;
    uint N = 10000000;
    uint Np = 2000*2000;
    umat coo(2,N);
    vec vals(N);
    //coo.set_size(2,N);
    //vals.set_size(N);
    for(uint i = 0; i < N; ++i){
        coo(0,i) = Np-i;
        coo(1,i) = 30;
        vals(i) = .01;
    }
    t0 = clock();
    M = sp_mat(coo, vals, Np,Np);
    cout << "a time:" <<  (float)(clock() - t0)/CLOCKS_PER_SEC << endl;
    */

    //arma::u32 sz = 600000000;
    //Eigen::VectorXcd vVppp(600000000);
    //vec vVppp = zeros(sz);
    //vec vVppp2(sz);
    //
    //double * aux_mem = new double[sz];
    //vec vVppp(aux_mem, sz, false, true);

    /*
    cout << vVppp.size() << endl;
    int e = 0;
    for(int i = 0; i<240000000; ++i){
        e += 1;
    }
    //delete aux_mem;
    */

    //cout << e << endl;

    //ccsolve solver2(fgas);
    //solver2.CCSD_SG(14);
    /*
    //testing the sparselibrary in Eigen
    typedef Eigen::Triplet<double> Tr;
    typedef Eigen::SparseMatrix<double> SparseMat;

    clock_t t1, t0, t2;
    t0 = clock();

    std::vector<Tr> tripletList;
    tripletList.reserve(vpppp.vValues.size());
    for(int i= 0; i<vpppp.vValues.size(); i++){
        tripletList.push_back(Tr(vpppp.vp(i)+vpppp.vq(i)*vpppp.iNp, vpppp.vr(i)+vpppp.vs(i)*vpppp.iNr, vpppp.vValues(i)));
    }
    SparseMat Vp1(vpppp.iNp*vpppp.iNq, vpppp.iNr*vpppp.iNs);
    Vp1.setFromTriplets(tripletList.begin(), tripletList.end());

    std::vector<Tr> tripletList2;
    tripletList2.reserve(vpphh.vValues.size());
    for(int i= 0; i<vpphh.vValues.size(); i++){
        tripletList2.push_back(Tr(vpphh.vp(i)+vpphh.vq(i)*vpphh.iNp, vpphh.vr(i)+vpphh.vs(i)*vpphh.iNr, vpphh.vValues(i)));
    }
    SparseMat Tp1(vpphh.iNp*vpphh.iNq, vpphh.iNr*vpphh.iNs);
    Tp1.setFromTriplets(tripletList2.begin(), tripletList2.end());
    t1 = clock();


    SparseMat Sp3 = Vp1*Tp1;
    t2 = clock();
    cout << "Eigen (Setup)         :" << t1-t0 << endl;
    cout << "Eigen (multiplication):" << t2-t1 << endl;
    cout << "Eigen (total)         :" << t2-t0 << endl;

    t0 = clock();
    sp_mat vp12 = vpppp.pq_rs();
    vp12 = vpphh.pq_rs();

    t1 = clock();
    vp12 = vpppp.pq_rs()*vpphh.pq_rs();
    t2 = clock();

    cout << "Armadillo (Setup)         :" << t1-t0 << endl;
    cout << "Armadillo (multiplication):" << t2-t1 << endl;
    cout << "Armadillo (total)         :" << t2-t0 << endl;

    */


    return 0;

}

