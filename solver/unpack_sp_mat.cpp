#include "unpack_sp_mat.h"

#define ARMA_64BIT_WORD
#include <armadillo>

using namespace std;
using namespace arma;

unpack_sp_mat::unpack_sp_mat(sp_mat c)
{
    int iNc = c.n_cols;
    int iN = 0;
    int iC = 0;
    int i,e;
    vT0.set_size(c.n_elem);
    cout << c.n_elem << " " << c.n_cols << " " << c.n_rows << endl;
    for(i= 0; i<c.n_cols; i++){
        iC = c.col_ptrs[i];

        //cout << iC << " " << c.row_indices[i] <<endl;
        /*
        for(e = 0; e<iC;e++){
            vVals(iN) = c.values[iN];
            vT0(iN) = c.row[0];
        }
        */
    }
}
