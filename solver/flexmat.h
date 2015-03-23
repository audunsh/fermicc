#ifndef FLEXMAT_H
#define FLEXMAT_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>

using namespace std;
using namespace arma;

class flexmat
{
public:
    //A class for flexible pphh-matrix manipulations
    flexmat(vec values, uvec a, uvec b, uvec i, uvec j, int Np, int Nh);
    int iNp;
    int iNp2;
    int iNh;
    int iNh2;
    int iNhp;
    int iNh2p;
    int iNp2h;


    vec vValues;
    uvec va;
    uvec vb;
    uvec vi;
    uvec vj;

    sp_mat smV;

    umat locations;

    //initialization budget

    //<pp||hh>
    int Nab_ij = 0;
    sp_mat Vab_ij;
    sp_mat ab_ij();

    int Nba_ij = 0;
    sp_mat Vba_ij;
    sp_mat ba_ij();

    int Nab_ji = 0;
    sp_mat Vab_ji;
    sp_mat ab_ji();

    int Nba_ji = 0;
    sp_mat Vba_ji;
    sp_mat ba_ji();

    // <ph||ph>
    int Nai_bj;
    sp_mat Vai_bj;
    sp_mat ai_bj();





};

#endif // FLEXMAT_H
