#include "unpack_sp_mat.h"

#define ARMA_64BIT_WORD
#include <armadillo>

using namespace std;
using namespace arma;

unpack_sp_mat::unpack_sp_mat(sp_mat c)
{
    int i,e;
    vT0.zeros(c.n_nonzero);
    vT1.zeros(c.n_nonzero);
    //vT0.set_size(c.n_nonzero);
    //vT1.set_size(c.n_nonzero);
    vVals.set_size(c.n_nonzero);

    for(i = 0; i<c.n_nonzero; i++){
        vVals(i) = c.values[i];
        vT0(i) = c.row_indices[i];
    }

    for(i= 0; i<c.n_cols; i++){
        int current_column = c.col_ptrs[i];
        int n_elem_in_column = c.col_ptrs[i+1]-c.col_ptrs[i];
        for(e = 0; e<n_elem_in_column; e++){
            vT1[current_column + e] = i;
        }
    }

}

