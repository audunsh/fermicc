#include "amplitude.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "basis/electrongas.h"

using namespace std;
using namespace arma;


amplitude::amplitude(electrongas bs, int n_configs)
{
    eBs = bs;
    k_step = 2*eBs.vKx.max()+3; //stepsize for identifying unique regions
    iNconfigs = n_configs;
    fmBlocks.set_size(iNconfigs); //block of indices
    fvConfigs.set_size(iNconfigs); //configuration in quantum numbers of each block
    blocklengths.set_size(iNconfigs);  //number of blocks in each configuration
    fmOrdering.set_size(iNconfigs); //the ordering of each configuration
    uiCurrent_block = 0;

    Np = eBs.iNbstates-eBs.iNparticles; //conflicting naming here
    Nh = eBs.iNparticles;

    uvSize.set_size(4); //particle-hole organization
    uvSize(0) = Np; //rows
    uvSize(1) = Np;
    uvSize(2) = Nh; //columns
    uvSize(3) = Nh;
}

// ##################################################
// ##                                              ##
// ## Element related functions                    ##
// ##                                              ##
// ##################################################

void amplitude::zeros(){} //zero out all elements
void amplitude::init_amplitudes(){} //initialize as amplitude
void amplitude::divide_energy(){} //divide all elements by corresponding energy (for amplitudes)

// ##################################################
// ##                                              ##
// ## Index related functions                      ##
// ##                                              ##
// ##################################################

ivec amplitude::intersect1d(ivec A, ivec B){
    // ###################################################
    // ## Returns the intersection of two unique arrays ##
    // ###################################################
    ivec ret(A.n_elem);
    uint counter = 0;
    for(int i = 0; i < A.n_elem; ++i){
        int a = A(i);
        for(int j = 0; j < B.n_elem; ++j){
            if(B(i) == a){
                ret(counter) = a;
                counter += 1;
                break;
            }
        }

    }
    ivec ret2(counter);
    for(int i = 0; i < counter; ++i){
        ret2(i) = ret(i);
    }
    return ret;
}

field<uvec> amplitude::unpack(uvec vStream, imat imOrder){
    // ###############################################
    // ## unpack a disorganized sequence of indices ##
    // ###############################################
    int iMsize = imOrder.n_cols+1;
    ivec M(iMsize);
    M(iMsize-1) = 1;
    int mn = 1;
    for(int i = 0; i < iMsize-1; ++i){
        mn *= uvSize(imOrder(i,0));
        M(iMsize - i-2) = mn;
    }
    field<uvec> indices(imOrder.n_cols);
    for(int i = 0; i < imOrder.n_cols; ++i){
        uvec P = vStream;
        for(int e = 0; e<i; ++e){
            P -= indices(e)*M(e+1);
        }
        indices(i) = floor(P/M(i+1));
    }
    return indices;
}

uvec amplitude::unpack_uvec(uint vStream, imat imOrder){
    // ###############################################
    // ## unpack a disorganized sequence of indices ##
    // ###############################################
    int iMsize = imOrder.n_cols+1;
    ivec M(iMsize);
    M(iMsize-1) = 1;
    int mn = 1;
    for(int i = 0; i < iMsize-1; ++i){
        mn *= uvSize(imOrder(i,0));
        M(iMsize - i-2) = mn;
    }
    uvec indices(imOrder.n_cols);
    for(int i = 0; i < imOrder.n_cols; ++i){
        uint P = vStream;
        for(int e = 0; e<i; ++e){
            P -= indices(e)*M(e+1);
        }
        indices(i) = floor(P/M(i+1));
    }
    return indices;
}



uint amplitude::to(uint p, uint q, uint r, uint s){
    return p + q*uvSize(0) + r*uvSize(0)*uvSize(1) + s * uvSize(0)*uvSize(1)*uvSize(2);
}  //compressed index

uvec amplitude::from(uint i){
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

void amplitude::map_regions(imat L, imat R){


    // ###########################################################
    // ## Counting number of rows and columns in representation ##
    // ###########################################################
    int iNrows = 1;
    int iNcols = 1;
    for(int i = 0; i<L.n_cols; ++i){
        cout << L(i,0) << endl;
        iNrows *= uvSize(L(i,0));
    }
    for(int i = 0; i<R.n_cols; ++i){
        iNcols *= uvSize(R(i,0));
    }

    // #########################################
    // ## Extract and organize actual indices ##
    // #########################################
    uvec rows = conv_to<uvec>::from(linspace(0,iNrows-1, iNrows));
    uvec cols = conv_to<uvec>::from(linspace(0,iNcols-1, iNcols));

    field<uvec> left = unpack(rows, L);
    field<uvec> right = unpack(cols, R);

    field<uvec> PQRS(4);
    for(int i = 0; i< left.n_rows; ++i){
        PQRS(L(i,0)) = left(i);
    }
    for(int i = 0; i< right.n_rows; ++i){
        PQRS(R(i,0)) = left(i);
    }

    // ############################################################
    // ## assign nonambiguous integer to each bra and ket config ##
    // ############################################################
    ivec LHS(iNrows); // = conv_to<ivec>::from(zeros(iNrows));
    ivec RHS(iNcols); // = conv_to<ivec>::from(zeros(iNcols));
    for(int i = 0; i<L.n_rows; ++i){
        LHS += eBs.unique(PQRS(L(i,0)+L(i,2)))*L(i,1);
    }
    for(int i = 0; i<R.n_rows; ++i){
        RHS += eBs.unique(PQRS(R(i,0)+R(i,2)))*R(i,1);
    }
    //RHS.print();

    // ####################################################################
    // ## Iterate over unique combinations, retain blocks where RHS==LHS ##
    // ####################################################################
    ivec unique_L = unique(LHS);
    ivec unique_R = unique(RHS);
    ivec K_unique = intersect1d(unique_L, unique_R);
    cout <<"K_unique: " << endl;
    K_unique.print();
    //unique_R.print();

    //ivec LHS_RHS = join_cols<imat>(RHS, LHS);
    //ivec K_unique = unique(LHS_RHS);

    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements;
    field<umat> tempBlockmap;

    for(uint i = 0; i<uiN; ++i){
        uvec row = rows.elem(find(LHS==K_unique(i)));
        uvec col = cols.elem(find(RHS==K_unique(i)));
        int Nx = row.n_elem;
        int Ny = col.n_elem;
        umat block(Nx,Ny);
        uvec pqrs(4);
        uvec tElements(Nx*Ny);
        umat tBlockmap(Nx*Ny, 3);
        uint index;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){
                uvec lhs = unpack_uvec(row(nx), L);
                uvec rhs = unpack_uvec(col(ny), R);
                for(uint j = 0; j<lhs.n_elem; ++j){
                    pqrs(L(j,0)) = lhs(j);
                }
                for(uint j = 0; j<rhs.n_elem; ++j){
                    pqrs(R(j,0)) = rhs(j);
                }
                index = to(pqrs(0), pqrs(1), pqrs(2), pqrs(3));

                tElements(nx*Ny + ny) = index;
                tBlockmap(nx*Ny + ny, 0) = i;
                tBlockmap(nx*Ny + ny, 1) = nx;
                tBlockmap(nx*Ny + ny, 2) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(uiCurrent_block)(i) = block;
        //block.print();
        cout << i << endl;



    }
    uiCurrent_block += 1;










} //map all regions defined by L == R


ivec amplitude::match_config(int u, ivec ivConfig){} //retrieve all
mat amplitude::getblock(int u, int i){}
mat amplitude::setblock(int u, int i, mat mBlock){}
mat amplitude::addblock(int u, int i, mat mBlock){}
