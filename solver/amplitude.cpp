#include "amplitude.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "basis/electrongas.h"

using namespace std;
using namespace arma;

amplitude::amplitude(){}

amplitude::amplitude(electrongas bs, int n_configs, uvec size)
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

    uvSize = size; //true state configurations (Np, Np, Nh, Nh) or (Np, Np, Np, Nh, Nh, Nh) (or similar)

    /*

    uvSize.set_size(4); //particle-hole organization
    uvSize(0) = Np; //rows
    uvSize(1) = Np;
    uvSize(2) = Nh; //columns
    uvSize(3) = Nh;
    */
}

void amplitude::init(electrongas bs, int n_configs, uvec size){
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

    uvSize = size; //true state configurations (Np, Np, Nh, Nh) or (Np, Np, Np, Nh, Nh, Nh) (or similar)

    /*

    uvSize.set_size(4); //particle-hole organization
    uvSize(0) = Np; //rows
    uvSize(1) = Np;
    uvSize(2) = Nh; //columns
    uvSize(3) = Nh;
    */
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

void amplitude::init_interaction(ivec shift){
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


    vElements.set_size(uvElements.n_rows);
    for(uint i= 0; i<uvElements.n_rows; ++i){
        uvec p = from(uvElements(i));
        //cout << p(0) <<  " " << p(1) << " " << p(2) << " " << p(3) << " "<< endl;
        vElements(i) = eBs.v2(p(0),p(1),p(2)+Nh,p(3)+Nh);
    }
}

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

uvec amplitude::unpack_uvec(uint vStream, imat imOrder){
    // ###############################################
    // ## unpack a disorganized sequence of indices ##
    // ###############################################
    uint iMsize = imOrder.n_rows+1;
    ivec M(iMsize);
    M(iMsize-1) = 1;
    uint mn = 1;
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



uint amplitude::to(uint p, uint q, uint r, uint s){
    return p + q*uvSize(0) + r*uvSize(0)*uvSize(1) + s * uvSize(0)*uvSize(1)*uvSize(2);
}  //compressed index

uvec amplitude::from(uint i){
    uvec ret(4);
    ret(3) = floor(i/(uvSize(0)*uvSize(1)*uvSize(2)));
    ret(2) = floor((i-ret(3)*uvSize(2)*uvSize(1)*uvSize(0))/(uvSize(0)*uvSize(1)));
    ret(1) = floor((i-ret(3)*uvSize(2)*uvSize(1)*uvSize(0) - ret(2)*uvSize(1)*uvSize(0))/uvSize(0));
    ret(0) = i-ret(3)*uvSize(2)*uvSize(1)*uvSize(0) - ret(2)*uvSize(1)*uvSize(0) - ret(1)*uvSize(0);
    return ret;
} //expanded index



uint amplitude::to6(uint p, uint q, uint r, uint s, uint t, uint u){
    return p + q*uvSize(0) + r*uvSize(0)*uvSize(1) + s * uvSize(0)*uvSize(1)*uvSize(2) + t*uvSize(0)*uvSize(1)*uvSize(2)*uvSize(4) + u*uvSize(0)*uvSize(1)*uvSize(2)*uvSize(4)*uvSize(5);
}  //compressed index, t3 amplitude

void amplitude::make_t3(){
    // ###########################################################
    // ## Make the tensor a t3 amplitude                        ##
    // ###########################################################
    uvSize = {Np, Np, Np, Nh, Nh, Nh};
    n5 = uvSize(0)*uvSize(1)*uvSize(2)*uvSize(3)*uvSize(4);
    n4 = uvSize(0)*uvSize(1)*uvSize(2)*uvSize(3);
    n3 = uvSize(0)*uvSize(1)*uvSize(2);
    n2 = uvSize(0)*uvSize(1);
    n1 = uvSize(0);
}


uvec amplitude::from6(uint i){
    uvec ret(6);
    ret(5) = floor(i/n5);
    ret(4) = floor((i-ret(5)*n5)/n4);
    ret(3) = floor((i-ret(5)*n5 - ret(4)*n4)/n3);
    ret(2) = floor((i-ret(5)*n5 - ret(4)*n4 - ret(3)*n3)/n2);
    ret(1) = floor((i-ret(5)*n5 - ret(4)*n4 - ret(3)*n3 - ret(2)*n2)/n1);
    ret(0) = i - ret(5)*n5 - ret(4)*n4 - ret(3)*n3 - ret(2)*n2 - ret(1)*n1;
    return ret;
} //expanded index, t3 amplitude



// ##################################################
// ##                                              ##
// ## External functions                           ##
// ##                                              ##
// ##################################################

void amplitude::map6(ivec left, ivec right){
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
        if(abs(left(i))<=3){
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
        if(abs(right(i))<=3){
            R(i,2) = Nh;
        }
        else{
            R(i,2) = 0;
        }
    }
    //R.print();
    map_regions6(L,R);

}

void amplitude::map(ivec left, ivec right){
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

field<uvec> amplitude::blocksort(ivec LHS, ivec K_unique){
    uvec l_sorted = sort_index(LHS);
    bool adv = false;

    //LHS.elem(l_sorted).print();
    //cout << endl;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);
    uvec row(100000);
    uint nx = 0;
    int l_c= LHS(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //first: align C and l_c
    //while(l_c<C){
    //    lc += 1;
    //    l_c = LHS(l_sorted(lc));
    //}


    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool br = false;
    bool row_collect = false;
    while(lc < uiS){
        l_c = LHS(l_sorted(lc));

        if(l_c == C){
            row(nx) = l_sorted(lc);
            nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }


        lc += 1;
        //cout << l_c - C << endl;
    }


    /*
    while(i<uiN){
        //while(l_c<C){
        //    lc += 1;
        //    l_c = LHS(l_sorted(lc));
        //}


        l_c = LHS(l_sorted(lc));

        adv = true;

        if(l_c == C){
            //add row index to block i
            row(nx) = l_sorted(lc);
            nx += 1;

            adv = false;
        }
        //lc += 1;


        if(adv){
            //advance to next uniquely defined block
            //append row to field
            tempRows(i) = row(span(0,nx)); //.print();
            //cout << endl;
            //cout << endl;
            i+=1;
            C = K_unique(i);
            nx = 0;
            //align l_c

            //l_c = LHS(l_sorted(lc));
            //while(l_c<C){
            //    lc += 1;
            //    l_c = LHS(l_sorted(lc));
            //}

        }
    }*/

    return tempRows;

}

void amplitude::map_regions6(imat L, imat R){


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


    field<uvec> PQRS(6);
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
    for(uint i = 0; i<iNrows;++i){
        LHS(i) = 0;
    }
    //ivec LHS = conv_to<ivec>::from(zeros(iNrows));

    ivec RHS(iNcols); // = conv_to<ivec>::from(zeros(iNcols));
    for(uint i = 0; i<iNcols;++i){
        RHS(i) = 0;
    }

    //LHS*=0;
    //RHS*=0;
    //ivec LHS = zeros<int> (iNrows);
    //cout << iNrows << " " << iNcols << endl;


    //cout << L.n_rows << " " << L.n_cols << " " << L.n_elem << endl;
    for(uint i = 0; i<L.n_rows; ++i){
        LHS += eBs.unique(PQRS(L(i,0))+L(i,2))*L(i,1);
        //LHS += eBs.unique(PQRS(L(i,0)));
    }
    for(uint i = 0; i<R.n_rows; ++i){
        RHS += eBs.unique(PQRS(R(i,0))+R(i,2))*R(i,1);
    }
    //cout << LHS.n_rows << endl;
    //RHS.print();

    // ####################################################################
    // ## Iterate over unique combinations, retain blocks where RHS==LHS ##
    // ####################################################################
    //LHS.print();
    //cout << LHS.n_elem << endl << endl;
    ivec unique_L = unique(LHS);
    ivec unique_R = unique(RHS);
    ivec K_unique = intersect1d(unique_L, unique_R);

    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;

    clock_t t0;

    t0 = clock();
    //experiment, trying to speed up initialization
    field<uvec> tempRows = blocksort(LHS, K_unique);
    field<uvec> tempCols = blocksort(RHS, K_unique);
    cout << " First method:" << (float)(clock()-t0)/CLOCKS_PER_SEC << endl;
    t0 = clock();


    for(uint i = 0; i<uiN; ++i){
        //uvec indx = find(LHS==K_unique(i));
        //LHS.elem(indx).print();
        //uvec row = rows.elem(find(LHS==K_unique(i)));
        //uvec col = cols.elem(find(RHS==K_unique(i)));
        uvec row = tempRows(i);
        uvec col = tempCols(i);
        //srow.print();
        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(6);
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
                index = to6(pqrs(0), pqrs(1), pqrs(2), pqrs(3), pqrs(4), pqrs(5));

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
    cout << "Second method:" << (float)(clock()-t0)/CLOCKS_PER_SEC << endl;
    t0 = clock();

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


    //if(all_resolved != true){
    if(tempN<tempL){
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }

        //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
        uvElements = join_cols<umat>(uvElements, remaining);
    }


    //lock and load
    uiCurrent_block += 1;










} //map all regions defined by L == R

void amplitude::map_regions(imat L, imat R){


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
    for(uint i = 0; i<iNrows;++i){
        LHS(i) = 0;
    }
    //ivec LHS = conv_to<ivec>::from(zeros(iNrows));

    ivec RHS(iNcols); // = conv_to<ivec>::from(zeros(iNcols));
    for(uint i = 0; i<iNcols;++i){
        RHS(i) = 0;
    }

    //LHS*=0;
    //RHS*=0;
    //ivec LHS = zeros<int> (iNrows);
    //cout << iNrows << " " << iNcols << endl;


    //cout << L.n_rows << " " << L.n_cols << " " << L.n_elem << endl;
    for(uint i = 0; i<L.n_rows; ++i){
        LHS += eBs.unique(PQRS(L(i,0))+L(i,2))*L(i,1);
        //LHS += eBs.unique(PQRS(L(i,0)));
    }
    for(uint i = 0; i<R.n_rows; ++i){
        RHS += eBs.unique(PQRS(R(i,0))+R(i,2))*R(i,1);
    }
    //cout << LHS.n_rows << endl;
    //RHS.print();

    // ####################################################################
    // ## Iterate over unique combinations, retain blocks where RHS==LHS ##
    // ####################################################################
    //LHS.print();
    //cout << LHS.n_elem << endl << endl;
    ivec unique_L = unique(LHS);
    ivec unique_R = unique(RHS);
    ivec K_unique = intersect1d(unique_L, unique_R);

    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;


    field<uvec> tempRows = blocksort(LHS, K_unique);
    field<uvec> tempCols = blocksort(RHS, K_unique);
    for(uint i = 0; i<uiN; ++i){
        //uvec indx = find(LHS==K_unique(i));
        //LHS.elem(indx).print();
        //uvec row = rows.elem(find(LHS==K_unique(i)));
        //uvec col = cols.elem(find(RHS==K_unique(i)));

        uvec row = tempRows(i);
        uvec col = tempCols(i);
        //cout << "row:" << endl;
        //row.print();
        //cout << "r1:" << endl;
        //row1.print();

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
    //cout << tempElementsSize << endl;

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


    //if(all_resolved != true){
    if(tempN<tempL){
        uvec remaining(tempL-tempN);
        uint tN = 0;
        while(tempN<tempL){
            remaining(tN) = flatElements(tempN);
            fmBlocks(uiCurrent_block)(flatBlockmap1(tempN))(flatBlockmap2(tempN),flatBlockmap3(tempN)) = trueN + tN;
            tempN += 1;
            tN += 1;
        }

        //cout << tempL << " "<< tempN << " " << remaining.n_elem << " " << uvElements.n_elem << endl;
        uvElements = join_cols<umat>(uvElements, remaining);
    }


    //lock and load
    uiCurrent_block += 1;
} //map all regions defined by L == R


mat amplitude::getblock(int u, int i){
    //umat block = fmBlocks(u)(i);
    mat block = vElements.elem(fmBlocks(u)(i));
    block.reshape(fmBlocks(u)(i).n_rows, fmBlocks(u)(i).n_cols);
    return block;
}

umat amplitude::getraw(int u, int i){
    //return index block, used for debugging and optimization
    return fmBlocks(u)(i);
}

void amplitude::setblock(int u, int i, mat mBlock){
    vElements.elem(fmBlocks(u)(i)) = vectorise(mBlock);
}

void amplitude::addblock(int u, int i, mat mBlock){
    //vectorise(mBlock);
    //cout << mBlock.size() << endl;
    //fmBlocks(u)(i).print();
    //cout << endl;
    //cout <<vElements.n_elem <<endl;
    //cout << endl;
    //vElements.elem(fmBlocks(u)(i)).print();
    //cout << endl;
    //mBlock.print();

    //cout << endl;
    //cout << endl;

    //vec elems = vElements.elem(fmBlocks(u)(i));
    //cout << elems.n_elem << endl;
    //cout << vElements.elem(fmBlocks(u)(i)).n_elem << endl;
    //vectorise(mBlock).print();

    vElements.elem(fmBlocks(u)(i)) += vectorise(mBlock);
}


void amplitude::compress(){
    //store only unique amplitude values
    //experimental functionality

    //this will probably not work (why?)
    //do instead: - treat each set of blocks distinctly, especially first set is easily compressible (store only first element in row)
    vec uniqueElements = unique(vElements);
    uvec tempind;
    for(uint i = 0; i < uniqueElements.n_rows; ++i){
        tempind = find(vElements==uniqueElements(i));

    }
}
