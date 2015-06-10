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

void amplitude::zeros(){
    vElements *= 0;
} //zero out all elements

void amplitude::init_amplitudes(){
    vElements.set_size(uvElements.n_rows);
    vEnergies.set_size(uvElements.n_rows);
    for(uint i= 0; i<uvElements.n_rows; ++i){
        uvec p = from(uvElements(i));
        //cout << p(0) <<  " " << p(1) << " " << p(2) << " " << p(3) << " "<< endl;
        vElements(i) = eBs.v2(p(0)+Nh,p(1)+Nh,p(2),p(3));
        //double v = eBs.vEnergy(p(2))+ eBs.vEnergy(p(3));
        vEnergies(i) = eBs.vEnergy(p(2)) + eBs.vEnergy(p(3))-eBs.vEnergy(p(0)+Nh)-eBs.vEnergy(p(1)+Nh);
    }
} //initialize as amplitude

void amplitude::divide_energy(){
    for(uint i= 0; i<uvElements.n_rows; ++i){
        vElements(i) /= vEnergies(i);
    }
} //divide all elements by corresponding energy (for amplitudes)

void amplitude::print_block_maximum(){
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

ivec amplitude::intersect1d(ivec A, ivec B){
    // ###################################################
    // ## Returns the intersection of two unique arrays ##
    // ###################################################
    ivec ret(A.n_rows);
    uint counter = 0;
    for(int i = 0; i < A.n_rows; ++i){
        int a = A(i);

        for(int j = 0; j < B.n_rows; ++j){
            if(B(j) == a){
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
    return ret2;
}

field<uvec> amplitude::unpack(uvec vStream, imat imOrder){
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
    for(int i = 0; i < imOrder.n_rows; ++i){
        uvec P = vStream;
        for(int e = 0; e<i; ++e){
            P -= indices(imOrder.n_rows-e-1)*M(e+1);
        }

        uvec elem = floor(P/M(i+1));
        indices(imOrder.n_rows-i-1) =  elem;
    }
    return indices;
}

uvec amplitude::unpack_uvec(uint vStream, imat imOrder){
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
    for(int i = 0; i < imOrder.n_rows; ++i){
        uint P = vStream;
        for(int e = 0; e<i; ++e){
            P -= indices(imOrder.n_rows - e - 1)*M(e+1);
        }
        indices(imOrder.n_rows - i - 1) = floor(P/M(i+1));
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

void amplitude::map(ivec left, ivec right){
    imat L(left.n_rows,3);
    for(uint i = 0; i<left.n_rows; ++i){
        L(i,0) =abs(left(i)) - 1;
        if(left(i)<0){
            L(i,1) = -1;
        }
        else{
            L(i,1) = 1;
        }
        if(abs(left(i))<=2){
            L(i,2) = Nh;
        }
        else{
            L(i,2) = 0;
        }
    }
    //L.print();

    imat R(right.n_rows,3);
    for(uint i = 0; i<right.n_rows; ++i){
        R(i,0) =abs(right(i)) - 1;
        if(right(i)<0){
            R(i,1) = -1;
        }
        else{
            R(i,1) = 1;
        }
        if(abs(right(i))<=2){
            R(i,2) = Nh;
        }
        else{
            R(i,2) = 0;
        }
    }
    //R.print();
    map_regions(L,R);

}

void amplitude::map_regions(imat L, imat R){


    // ###########################################################
    // ## Counting number of rows and columns in representation ##
    // ###########################################################
    int iNrows = 1;
    int iNcols = 1;
    for(int i = 0; i<L.n_rows; ++i){
        //cout << L(i,0) << endl;
        iNrows *= uvSize(L(i,0));
    }
    for(int i = 0; i<R.n_rows; ++i){
        iNcols *= uvSize(R(i,0));
    }

    // #########################################
    // ## Extract and organize actual indices ##
    // #########################################
    uvec rows = conv_to<uvec>::from(linspace(0,iNrows-1, iNrows));
    uvec cols = conv_to<uvec>::from(linspace(0,iNcols-1, iNcols));

    field<uvec> left = unpack(rows, L);
    field<uvec> right = unpack(cols, R);
    //left.print();
    //cout << "rows:" << left.n_rows << endl;

    field<uvec> PQRS(4);
    for(int i = 0; i< left.n_rows; ++i){
        PQRS(L(i,0)) = left(i);
    }
    for(int i = 0; i< right.n_rows; ++i){
        PQRS(R(i,0)) = right(i);
    }
    //PQRS.print();

    // ############################################################
    // ## assign nonambiguous integer to each bra and ket config ##
    // ############################################################
    ivec LHS(iNrows); // = conv_to<ivec>::from(zeros(iNrows));
    ivec RHS(iNcols); // = conv_to<ivec>::from(zeros(iNcols));
    LHS*=0;
    RHS*=0;
    //cout << L.n_rows << " " << L.n_cols << " " << L.n_elem << endl;
    for(int i = 0; i<L.n_rows; ++i){
        LHS += eBs.unique(PQRS(L(i,0))+L(i,2))*L(i,1);
        //LHS += eBs.unique(PQRS(L(i,0)));
    }
    for(int i = 0; i<R.n_rows; ++i){
        RHS += eBs.unique(PQRS(R(i,0))+R(i,2))*R(i,1);
    }
    //cout << LHS.n_rows << endl;
    //RHS.print();

    // ####################################################################
    // ## Iterate over unique combinations, retain blocks where RHS==LHS ##
    // ####################################################################
    ivec unique_L = unique(LHS);
    ivec unique_R = unique(RHS);
    ivec K_unique = intersect1d(unique_L, unique_R);
    //cout <<"K_unique: " << endl;
    //K_unique.print();
    //unique_R.print();

    //ivec LHS_RHS = join_cols<imat>(RHS, LHS);
    //ivec K_unique = unique(LHS_RHS);

    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;

    for(uint i = 0; i<uiN; ++i){
        //uvec indx = find(LHS==K_unique(i));
        //LHS.elem(indx).print();
        uvec row = rows.elem(find(LHS==K_unique(i)));
        uvec col = cols.elem(find(RHS==K_unique(i)));
        //srow.print();
        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(4);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        uint index;
        tempElementsSize += Nx*Ny;
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
                tBlockmap1(nx*Ny + ny) = i;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(uiCurrent_block)(i) = block;
        tempElements(i) = tElements;
        tempBlockmap1(i) = tBlockmap1;
        tempBlockmap2(i) = tBlockmap2;
        tempBlockmap3(i) = tBlockmap3;

        //block.print();
        //cout << i << endl;
    }
    //fmBlocks(uiCurrent_block)(3).print();


    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    uint counter = 0;
    for(uint i = 0; i<uiN; ++i){
        for(uint j = 0; j < tempElements(i).n_rows; ++j){
            flatElements(counter) = tempElements(i)(j);
            flatBlockmap1(counter) = tempBlockmap1(i)(j);
            flatBlockmap2(counter) = tempBlockmap2(i)(j);
            flatBlockmap3(counter) = tempBlockmap3(i)(j);
            counter += 1;
        }
        tempElements(i).set_size(0);
        tempBlockmap1(i).set_size(0);
        tempBlockmap2(i).set_size(0);
        tempBlockmap3(i).set_size(0);

    }

    // ####################################################################
    // ## Consolidate blocks with existing configurations                ##
    // ####################################################################


    umat n = sort_index(flatElements);
    flatElements = flatElements.elem(n);
    flatBlockmap1 = flatBlockmap1.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap2 = flatBlockmap2.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,
    flatBlockmap3 = flatBlockmap3.elem(n); //DOES THIS SORT PROPERLY? UNKNOWN,

    uint tempN = 0;
    uint trueN = 0;
    uint tempL = flatElements.n_rows;
    uint trueL = uvElements.n_rows;
    bool all_resolved = false;
    while(trueN < trueL){
        if(uvElements(trueN) == flatElements(tempN)){
            //identical indexes occuring in
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN;
            trueN += 1;
            tempN += 1;
        }
        else{
            if(uvElements(trueN) == flatElements(tempN)){
                trueN += 1;
            }
            else{
                tempN += 1;
                if(tempN>=tempL){
                    all_resolved = true;
                    break;
                }
            }

        }
    }


    if(all_resolved != true){
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }
        uvElements = join_cols<umat>(uvElements, remaining);
    }


    //lock and load
    uiCurrent_block += 1;










} //map all regions defined by L == R


ivec amplitude::match_config(int u, ivec ivConfig){} //retrieve all
mat amplitude::getblock(int u, int i){
    //umat block = fmBlocks(u)(i);
    mat block = vElements.elem(fmBlocks(u)(i));
    block.reshape(fmBlocks(u)(i).n_rows, fmBlocks(u)(i).n_cols);
    return block;
}
mat amplitude::setblock(int u, int i, mat mBlock){}
mat amplitude::addblock(int u, int i, mat mBlock){}
