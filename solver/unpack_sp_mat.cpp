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
    int iCN = 0;
    int i,e;
    vT0.set_size(c.n_nonzero);
    vT1.set_size(c.n_nonzero);
    //mLocations.set_size(c.n_elem,2);
    //locations.col(0) = vp + vq*iNp;
    //locations.col(1) = vr + vs*iNr;

    vVals.set_size(c.n_nonzero);
    //cout << c.n_elem << " " << c.n_cols << " " << c.n_rows << endl;
    /*
    for(i = 0; i<c.n_elem; i++){
        vT0(i) = c.row_indices[i];
        vVals(i) = c.values[i];
        //vT1(i) = ?
    }
    */
    for(i = 0; i<c.n_nonzero; i++){
        vVals(i) = c.values[i];
        vT0(i) = c.row_indices[i];
    }



    for(i= 0; i<c.n_cols-1; i++){
        //vT1.elem(conv_to<uvec>::from(span(c.col_ptrs[i], c.col_ptrs[i+1]))) = i;

        int current_column = c.col_ptrs[i];
        int n_elem_in_column = c.col_ptrs[i+1]-c.col_ptrs[i];
        for(e = 0; e<n_elem_in_column; e++){
            //vVals[current_column + e] = c.values[current_column + e];
            //vT0[current_column + e] = c.row_indices[current_column + e];
            vT1[current_column + e] = i; //current_column;

        }



        /*

        iC = c.col_ptrs[i];

        iCN = c.col_ptrs[i+1];

        //cout << iC << " " << c.row_indices[i] <<endl;

        for(e = 0; e<iCN-iC;e++){
            vVals(iN) = c.values[iC + e];
            vT0(iN) = c.row_indices[iC + e];
            vT1(iN) = i;
            //vT0(iN) = c.row_indices[iN];
            //vT1(iN) = i;
            iN += 1;
        }*/


    }
    //cout << c.n_cols << " " << c.n_rows <<  endl;
    //cout << c.begin_col() << endl;
    //cout << iN << endl;
    //cout << c.n_nonzero << endl;
    //for(i = 0; i<c.n_nonzero; i++){
    //    cout << vT0(i) << " " << vT1(i) << endl;
    //}
    //vT0.print();
    //vT1.print();

}
