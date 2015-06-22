#include "blockmap.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "basis/electrongas.h"

using namespace std;
using namespace arma;

blockmap::blockmap(){}

void blockmap::init(electrongas bs, int n_configs, uvec size){
    eBs = bs;
    k_step = 2*eBs.vKx.max()+3; //stepsize for identifying unique regions
    iNconfigs = n_configs;
    fmBlocks.set_size(iNconfigs); //block of indices

    fmBlockz.set_size(iNconfigs);

    fuvRows.set_size(iNconfigs);
    fuvCols.set_size(iNconfigs);



    fvConfigs.set_size(iNconfigs); //configuration in quantum numbers of each block
    blocklengths.set_size(iNconfigs);  //number of blocks in each configuration
    fmOrdering.set_size(iNconfigs,2); //the ordering of each configuration
    uiCurrent_block = 0;

    Np = eBs.iNbstates-eBs.iNparticles; //conflicting naming here
    Nh = eBs.iNparticles;

    uvSize = size; //true state configurations (Np, Np, Nh, Nh) or (Np, Np, Np, Nh, Nh, Nh) (or similar)

    /*

    uvSize.set_size(4); //particle-hole organization
    uvSize(0) = Np; //rows
    uvSize(1) = Np;
    uvSize(2) = Nh; //columns
    uvSize(3) = Nh;
    */
}

blockmap::blockmap(electrongas bs, int n_configs, uvec size)
{
    eBs = bs;
    k_step = 2*eBs.vKx.max()+3; //stepsize for identifying unique regions
    iNconfigs = n_configs;
    fmBlocks.set_size(iNconfigs); //block of indices

    fmBlockz.set_size(iNconfigs);

    fuvRows.set_size(iNconfigs);
    fuvCols.set_size(iNconfigs);

    fvConfigs.set_size(iNconfigs); //configuration in quantum numbers of each block
    blocklengths.set_size(iNconfigs);  //number of blocks in each configuration
    fmOrdering.set_size(iNconfigs,2); //the ordering of each configuration
    uiCurrent_block = 0;

    Np = eBs.iNbstates-eBs.iNparticles; //conflicting naming here
    Nh = eBs.iNparticles;

    uvSize = size; //true state configurations (Np, Np, Nh, Nh) or (Np, Np, Np, Nh, Nh, Nh) (or similar)

    /*

    uvSize.set_size(4); //particle-hole organization
    uvSize(0) = Np; //rows
    uvSize(1) = Np;
    uvSize(2) = Nh; //columns
    uvSize(3) = Nh;
    */
}



void blockmap::init_interaction(ivec shift){
    //imat L(2,3);
    //imat R(2,3);

    ivec l0 = {0,1,shift(0)};
    ivec l1 = {1,1,shift(1)};
    ivec r0 = {2,1,shift(2)};
    ivec r1 = {3,1,shift(3)};

    imat L = join_rows(l0,l1);
    imat R = join_rows(r0,r1);
    //L.print();
    map_regions(L.t(), R.t());
}

void blockmap::print_block_maximum(){
    //debugging function
    for(uint i = 0; i < blocklengths(0); ++i){
        getblock(0, i).print();
        cout << endl;
    }
}

// ##################################################
// ##                                              ##
// ## Index related functions                      ##
// ##                                              ##
// ##################################################

ivec blockmap::intersect1d(ivec A, ivec B){
    // ###################################################
    // ## Returns the intersection of two unique arrays ##
    // ###################################################
    ivec ret(A.n_rows);
    uint counter = 0;
    for(uint i = 0; i < A.n_rows; ++i){
        int a = A(i);

        for(uint j = 0; j < B.n_rows; ++j){
            if(B(j) == a){
                ret(counter) = a;
                counter += 1;
                break;
            }
        }

    }
    ivec ret2(counter);
    for(uint i = 0; i < counter; ++i){
        ret2(i) = ret(i);
    }
    return ret2;
}

field<uvec> blockmap::unpack(uvec vStream, imat imOrder){
    // ###############################################
    // ## unpack a disorganized sequence of indices ##
    // ###############################################
    int iMsize = imOrder.n_rows+1;
    ivec M(iMsize);
    M(iMsize-1) = 1;
    int mn = 1;
    for(int i = 0; i < iMsize-1; ++i){
        mn *= uvSize(imOrder(i,0));
        M(iMsize - i-2) = mn;
    }
    field<uvec> indices(imOrder.n_rows);
    for(uint i = 0; i < imOrder.n_rows; ++i){
        uvec P = vStream;
        for(uint e = 0; e<i; ++e){
            P -= indices(imOrder.n_rows-e-1)*M(e+1);
        }

        uvec elem = floor(P/M(i+1));
        indices(imOrder.n_rows-i-1) =  elem;
    }
    return indices;
}

uvec blockmap::unpack_uvec(uint vStream, imat imOrder){
    // ###############################################
    // ## unpack a disorganized sequence of indices ##
    // ###############################################
    int iMsize = imOrder.n_rows+1;
    ivec M(iMsize);
    M(iMsize-1) = 1;
    int mn = 1;
    for(int i = 0; i < iMsize-1; ++i){
        mn *= uvSize(imOrder(i,0));
        M(iMsize - i-2) = mn;
    }
    uvec indices(imOrder.n_rows);
    for(uint i = 0; i < imOrder.n_rows; ++i){
        uint P = vStream;
        for(uint e = 0; e<i; ++e){
            P -= indices(imOrder.n_rows - e - 1)*M(e+1);
        }
        indices(imOrder.n_rows - i - 1) = floor(P/M(i+1));
    }
    return indices;
}



uint blockmap::to(uint p, uint q, uint r, uint s){
    return p + q*uvSize(0) + r*uvSize(0)*uvSize(1) + s * uvSize(0)*uvSize(1)*uvSize(2);
}  //compressed index

uvec blockmap::from(uint i){
    uvec ret(4);
    ret(3) = floor(i/(uvSize(0)*uvSize(1)*uvSize(2)));
    ret(2) = floor((i-ret(3)*uvSize(0)*uvSize(1)*uvSize(2))/(uvSize(0)*uvSize(1)));
    ret(1) = floor((i-ret(3)*uvSize(0)*uvSize(1)*uvSize(2) - ret(2)*uvSize(0)*uvSize(1))/uvSize(0));
    ret(0) = i-ret(3)*uvSize(0)*uvSize(1)*uvSize(2) - ret(2)*uvSize(0)*uvSize(1) - ret(1)*uvSize(0);
    return ret;
} //expanded index

// ##################################################
// ##                                              ##
// ## External functions                           ##
// ##                                              ##
// ##################################################

void blockmap::map(ivec left, ivec right){
    imat L(left.n_rows,3);
    //left.print();
    for(uint i = 0; i<left.n_rows; ++i){
        L(i,0) =abs(left(i)) - 1;
        if(left(i)<0){
            L(i,1) = -1;
        }
        else{
            L(i,1) = 1;
        }



        if(uvSize(abs(left(i))-1)==Np){
            L(i,2) = Nh;
        }
        else{
            L(i,2) = 0;
        }
    }

    imat R(right.n_rows,3);
    for(uint i = 0; i<right.n_rows; ++i){
        R(i,0) =abs(right(i)) - 1;
        if(right(i)<0){
            R(i,1) = -1;
        }
        else{
            R(i,1) = 1;
        }
        if(uvSize(abs(right(i))-1)==Np){
            R(i,2) = Nh;
        }
        else{
            R(i,2) = 0;
        }
    }
    map_regions(L,R);

}

void blockmap::map_vpppp(){
    // ###########################################################
    // ## Special care given to the ladder term                 ##
    // ###########################################################
    fmOrdering(uiCurrent_block,0) = L;
    fmOrdering(uiCurrent_block,1) = R;

    uint iNp = uvSize(0);
    uint iNp2 = uvSize(0)*uvSize(0);
    vec AB = linspace(0,iNp2-1,iNp2);

    uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    uvec A = conv_to<uvec>::from(AB) - B*iNp;

    ivec KAB = eBs.unique(A+Nh) + eBs.uniqe(B+Nh);

    ivec K_unique = unique(KAB);

    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    //fmBlocks(uiCurrent_block).set_size(uiN);
    fmBlockz(uiCurrent_block).set_size(uiN,2);
    fuvCols(uiCurrent_block).set_size(uiN);
    fuvRows(uiCurrent_block).set_size(uiN);

    //field<uvec> tempElements(uiN);
    //field<uvec> tempBlockmap1(uiN);
    //field<uvec> tempBlockmap2(uiN);
    //field<uvec> tempBlockmap3(uiN);

    //uint tempElementsSize = 0;

    for(uint i = 0; i<uiN; ++i){
        uvec row = AB.elem(find(KAB==K_unique(i)));

        fmBlockz(uiCurrent_block)(i,0) = row;
        fmBlockz(uiCurrent_block)(i,1) = row;

        fuvRows(uiCurrent_block)(i) = row;
        fuvCols(uiCurrent_block)(i) = row;


    }

    //lock and load
    uiCurrent_block += 1;




}

void blockmap::map_regions(imat L, imat R){

    fmOrdering(uiCurrent_block,0) = L;
    fmOrdering(uiCurrent_block,1) = R;



    // ###########################################################
    // ## Counting number of rows and columns in representation ##
    // ###########################################################
    uint iNrows = 1;
    uint iNcols = 1;

    for(uint i = 0; i<L.n_rows; ++i){
        iNrows *= uvSize(L(i,0));
    }
    for(uint i = 0; i<R.n_rows; ++i){
        iNcols *= uvSize(R(i,0));
    }

    // #########################################
    // ## Extract and organize actual indices ##
    // #########################################
    //uvec rows = conv_to<uvec>::from(linspace(0,iNrows-1, iNrows));

    uvec rows = linspace<uvec>(0,iNrows-1, iNrows);
    uvec cols = linspace<uvec>(0,iNcols-1, iNcols);

    //uvec cols = conv_to<uvec>::from(linspace(0,iNcols-1, iNcols));

    field<uvec> left = unpack(rows, L);
    field<uvec> right = unpack(cols, R);


    field<uvec> PQRS(4);
    for(uint i = 0; i< left.n_rows; ++i){
        PQRS(L(i,0)) = left(i);
    }
    for(uint i = 0; i< right.n_rows; ++i){
        PQRS(R(i,0)) = right(i);
    }
    //PQRS.print();

    // ############################################################
    // ## assign nonambiguous integer to each bra and ket config ##
    // ############################################################
    ivec LHS(iNrows); // = conv_to<ivec>::from(zeros(iNrows));
    for(uint i = 0; i < iNrows; ++i){
        LHS(i) = 0;
    }
    ivec RHS(iNcols); // = conv_to<ivec>::from(zeros(iNcols));
    for(uint i = 0; i < iNcols; ++i){
        RHS(i) = 0;
    }

    for(uint i = 0; i<L.n_rows; ++i){
        LHS += eBs.unique(PQRS(L(i,0))+L(i,2))*L(i,1);
    }
    for(uint i = 0; i<R.n_rows; ++i){
        RHS += eBs.unique(PQRS(R(i,0))+R(i,2))*R(i,1);
    }


    // ####################################################################
    // ## Iterate over unique combinations, retain blocks where RHS==LHS ##
    // ####################################################################
    ivec unique_L = unique(LHS);
    ivec unique_R = unique(RHS);
    ivec K_unique = intersect1d(unique_L, unique_R);

    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    //fmBlocks(uiCurrent_block).set_size(uiN);
    fmBlockz(uiCurrent_block).set_size(uiN,2);
    fuvCols(uiCurrent_block).set_size(uiN);
    fuvRows(uiCurrent_block).set_size(uiN);


    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    //uint tempElementsSize = 0;

    for(uint i = 0; i<uiN; ++i){
        uvec row = rows.elem(find(LHS==K_unique(i)));
        uvec col = cols.elem(find(RHS==K_unique(i)));

        fmBlockz(uiCurrent_block)(i,0) = row;
        fmBlockz(uiCurrent_block)(i,1) = col;

        fuvRows(uiCurrent_block)(i) = row;
        fuvCols(uiCurrent_block)(i) = col;


    }

    //lock and load
    uiCurrent_block += 1;










} //map all regions defined by L == R

mat blockmap::getblock_vpppp(int u, int i){

}

mat blockmap::getblock(int u, int i){
    //umat block = fmBlocks(u)(i);
    //mat block = vElements.elem(fmBlocks(u)(i));
    //block.reshape(fmBlocks(u)(i).n_rows, fmBlocks(u)(i).n_cols);
    //return block;

    uvec row = fmBlockz(u)(i,0);
    uvec col = fmBlockz(u)(i,1);

    imat L = fmOrdering(u,0);
    imat R = fmOrdering(u,1);

    int Nx = row.n_rows;
    int Ny = col.n_rows;
    //cout << Nx << " " << Ny << " " << endl; //" " << K_unique(i) << endl;
    mat block(Nx,Ny);


    uvec pqrs(4);
    for(int nx = 0; nx < Nx; nx++){
        for(int ny = 0; ny < Ny; ny++){
            uvec lhs = unpack_uvec(row(nx), L);
            uvec rhs = unpack_uvec(col(ny), R);
            for(uint j = 0; j<lhs.n_elem; ++j){
                pqrs(L(j,0)) = lhs(j);
                if(uvSize(L(j,0))==Np){
                    pqrs(L(j,0)) += Nh;
                }
            }
            for(uint j = 0; j<rhs.n_elem; ++j){
                pqrs(R(j,0)) = rhs(j);
                if(uvSize(R(j,0))==Np){
                    pqrs(R(j,0)) += Nh;
                }
            }
            //index = to(pqrs(0), pqrs(1), pqrs(2), pqrs(3));
            block(nx, ny) = eBs.v2(pqrs(0), pqrs(1), pqrs(2), pqrs(3)); //remember to add in the needed shifts
        }
    }
    //cout << Nx << " " << Ny << " " << endl; //" " << K_unique(i) << endl;
    return block;

}

