#include "flexmat.h"

flexmat::flexmat(sp_mat mV, int Np, int Nh)
{
    iNp = Np;
    iNh = Nh;
    smV = mV;
}

sp_mat flexmat::ab_ij(){
    return smV;
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
