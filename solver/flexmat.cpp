#include "flexmat.h"
#define ARMA_64BIT_WORD
#include <armadillo>

using namespace std;
using namespace arma;


flexmat::flexmat(vec values, uvec a, uvec b, uvec i, uvec j, int Np, int Nh)
{
    iNp = Np;
    iNh = Nh;
    iNp2 = Np*Np;
    iNh2 = Nh*Nh;
    iNhp = Nh*Np;
    iNp2h = iNp2*iNh;
    iNh2p = iNh2*iNp;

    vValues = values;
    va = a;
    vb = b;
    vi = i;
    vj = j;

}

sp_mat flexmat::ab_ij(){
    if(Nab_ij == 0){
        locations.set_size(va.size(),2);
        locations.col(0) = va + vb*iNp;
        locations.col(1) = vi + vj*iNh;
        Vab_ij = sp_mat(locations.t(), vValues, iNp2, iNh2);
        Nab_ij = 1;
        return Vab_ij;
    }
    else{
        return Vab_ij;
    }
}

sp_mat flexmat::ba_ij(){
    if(Nba_ij == 0){
        locations.set_size(va.size(),2);
        locations.col(0) = vb + va*iNp;
        locations.col(1) = vi + vj*iNh;
        Vba_ij = sp_mat(locations.t(), vValues, iNp2, iNh2);
        Nba_ij = 1;
        return Vba_ij;
    }
    else{
        return Vba_ij;
    }
}

sp_mat flexmat::ab_ji(){
    if(Nab_ji == 0){
        locations.set_size(va.size(),2);
        locations.col(0) = va + vb*iNp;
        locations.col(1) = vj + vi*iNh;
        Vab_ji = sp_mat(locations.t(), vValues, iNp2, iNh2);
        Nab_ji = 1;
        return Vbab_ji;
    }
    else{
        return Vab_ji;
    }
}

sp_mat flexmat::ba_ji(){
    if(Nba_ji == 0){
        locations.set_size(va.size(),2);
        locations.col(0) = vb + va*iNp;
        locations.col(1) = vj + vi*iNh;
        Vba_ji = sp_mat(locations.t(), vValues, iNp2, iNh2);
        Nba_ji = 1;
        return Vba_ji;
    }
    else{
        return Vba_ji;
    }
}


sp_mat flexmat::ai_bj(){
    //cout << smV.col_ptrs << endl;
    smV.vec_state;
    cout << smV.vec_state << endl;
    cout << smV.row_indices[0] << endl;
    //cout << smV.values << endl;
    //smV.row_indices

    return smV;
}
