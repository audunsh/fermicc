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

        //Vpppp = sp_mat(locations.t(), values, iNp2, iNp2);

        Nab_ij = 1;
        return Vab_ij;
    }
    else{
        return Vab_ij;
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
