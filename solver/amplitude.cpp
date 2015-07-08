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
    permutative_ordering.set_size(iNconfigs);
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

    permutative_ordering.set_size(iNconfigs);

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
    //uvElements.print(); //could we maybe retrieve these "on the fly" ? (would mean to locate blocks "on the fly")
    for(uint i= 0; i<uvElements.n_rows; ++i){
        uvec p = from(uvElements(i));
        //cout << p(0) <<  " " << p(1) << " " << p(2) << " " << p(3) << " "<< endl;
        vElements(i) = eBs.v2(p(0)+Nh,p(1)+Nh,p(2),p(3));
        //double v = eBs.vEnergy(p(2))+ eBs.vEnergy(p(3));
        //vEnergies(i) = eBs.vEnergy(p(2)) + eBs.vEnergy(p(3))-eBs.vEnergy(p(0)+Nh)-eBs.vEnergy(p(1)+Nh);

        vEnergies(i) = eBs.F(p(2)) + eBs.F(p(3))-eBs.F(p(0)+Nh)-eBs.F(p(1)+Nh);
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
    // #####################################################
    // ## Sort indices of LHS into corresponding blocks   ##
    // #####################################################


    uvec l_sorted = sort_index(LHS); //most time is spent doing this -- maybe we should reorganize the items so they are sorted all along?
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


void amplitude::map_t2_permutations(){
    // ###################################################################
    // ## Set up amplitude as t3temp with index permutations in blocks  ##
    // ###################################################################

    //Basically, we set up the standard amplitude sorting as abc-ijk, but store dimensions of each block so we ,may easily permute them later
    field<ivec> ab = pp();
    field<ivec> ij = hh_compact();

    ivec K_unique = intersect1d(unique(ab(2)), unique(ij(2)));
    field<uvec> tempRows = partition_pp_permutations(ab, K_unique);
    //field<uvec> tempCols = partition(ijk(3), K_unique);
    field<uvec> tempCols = partition_hh_permutations(ij, K_unique);
    uvec row;
    uvec col;
    uvec a,b,c;
    uvec i,j,k;


    //for use in actual amplitude mapping
    permutative_ordering.set_size(K_unique.n_rows);
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;
    //uvec a,b,c,i,j,k;
    uint systemsize = 0;
    for(uint n = 0; n <K_unique.n_rows; ++n){
        uvec dim(6);
        row = tempRows(n);
        col = tempCols(n);
        //systemsize += row.n_rows*col.n_rows;


        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(4);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        b = floor(row/Np); //k
        a = row  - b*Np;

        j = floor(col/Nh); //k

        i = col  - j*Nh;


        uint index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){

                index = to(a(nx), b(nx), i(ny), j(ny));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = n;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(0)(n) = block;
        tempElements(n) = tElements;
        tempBlockmap1(n) = tBlockmap1; //block that element belongs to
        tempBlockmap2(n) = tBlockmap2; //row of element
        tempBlockmap3(n) = tBlockmap3; //column of element








    }



    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){
            flatElements(counter) = tempElements(ni)(nj);
            flatBlockmap1(counter) = tempBlockmap1(ni)(nj);
            flatBlockmap2(counter) = tempBlockmap2(ni)(nj);
            flatBlockmap3(counter) = tempBlockmap3(ni)(nj);
            counter += 1;
        }
        tempElements(ni).set_size(0);
        tempBlockmap1(ni).set_size(0);
        tempBlockmap2(ni).set_size(0);
        tempBlockmap3(ni).set_size(0);

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

}


void amplitude::map_t3_permutations(){
    // ###################################################################
    // ## Set up amplitude as t3temp with index permutations in blocks  ##
    // ###################################################################

    //Basically, we set up the standard amplitude sorting as abc-ijk, but store dimensions of each block so we ,may easily permute them later
    field<ivec> abc = ppp({1,1,1});
    field<ivec> ijk = hhh();

    ivec K_unique = intersect1d(unique(abc(3)), unique(ijk(3)));
    field<uvec> tempRows = partition_ppp_permutations(abc, K_unique);
    //field<uvec> tempCols = partition(ijk(3), K_unique);
    field<uvec> tempCols = partition_hhh_permutations(ijk, K_unique);
    uvec row;
    uvec col;
    uvec a,b,c;
    uvec i,j,k;


    //for use in actual amplitude mapping
    permutative_ordering.set_size(K_unique.n_rows);
    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    fmBlocks(uiCurrent_block).set_size(uiN);

    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    uint tempElementsSize = 0;
    //uvec a,b,c,i,j,k;
    uint systemsize = 0;
    for(uint n = 0; n <K_unique.n_rows; ++n){
        uvec dim(6);
        row = tempRows(n);
        col = tempCols(n);
        //systemsize += row.n_rows*col.n_rows;


        int Nx = row.n_rows;
        int Ny = col.n_rows;
        //cout << Nx << " " << Ny << " " << " " << K_unique(i) << endl;
        umat block(Nx,Ny);
        uvec pqrs(4);
        uvec tElements(Nx*Ny);
        uvec tBlockmap1(Nx*Ny);
        uvec tBlockmap2(Nx*Ny);
        uvec tBlockmap3(Nx*Ny);

        c = floor(row/(Np*Np)); //k
        b = floor(((row - c*Np*Np))/Np);
        a = row - c*Np*Np - b*Np;

        k = floor(col/(Nh*Nh)); //k
        j = floor(((col - k*Nh*Nh))/Nh);
        i = col - k*Nh*Nh - j*Nh;


        uint index;
        tempElementsSize += Nx*Ny;
        for(int nx = 0; nx < Nx; nx++){
            for(int ny = 0; ny < Ny; ny++){

                index = to6(a(nx), b(nx), c(nx), i(ny), j(ny), k(ny));

                tElements(nx*Ny + ny) = index;
                tBlockmap1(nx*Ny + ny) = n;
                tBlockmap2(nx*Ny + ny) = nx;
                tBlockmap3(nx*Ny + ny) = ny;
                block(nx, ny) = index;
            }
        }
        fmBlocks(0)(n) = block;
        tempElements(n) = tElements;
        tempBlockmap1(n) = tBlockmap1; //block that element belongs to
        tempBlockmap2(n) = tBlockmap2; //row of element
        tempBlockmap3(n) = tBlockmap3; //column of element








    }
    cout << "Blocks:" << tempRows.n_rows << endl;
    cout << "Size:" << systemsize << endl;
    cout << "init_size:" << Np*(Np+1)*(Np+2)/6 << endl;


    // ####################################################################
    // ## Flatten tempElements and tempBlockmap                          ##
    // ####################################################################
    uvec flatElements(tempElementsSize);
    uvec flatBlockmap1(tempElementsSize);
    uvec flatBlockmap2(tempElementsSize);
    uvec flatBlockmap3(tempElementsSize);

    uint counter = 0;
    for(uint ni = 0; ni<uiN; ++ni){
        for(uint nj = 0; nj < tempElements(ni).n_rows; ++nj){
            flatElements(counter) = tempElements(ni)(nj);
            flatBlockmap1(counter) = tempBlockmap1(ni)(nj);
            flatBlockmap2(counter) = tempBlockmap2(ni)(nj);
            flatBlockmap3(counter) = tempBlockmap3(ni)(nj);
            counter += 1;
        }
        tempElements(ni).set_size(0);
        tempBlockmap1(ni).set_size(0);
        tempBlockmap2(ni).set_size(0);
        tempBlockmap3(ni).set_size(0);

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




}

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

    //field<uvec> tempRows = blocksort(LHS, K_unique);
    //field<uvec> tempCols = blocksort(RHS, K_unique);
    for(uint i = 0; i<uiN; ++i){
        uvec indx = find(LHS==K_unique(i));
        //LHS.elem(indx).print();
        uvec row = rows.elem(find(LHS==K_unique(i)));
        uvec col = cols.elem(find(RHS==K_unique(i)));





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
        tempBlockmap1(i) = tBlockmap1; //block that element belongs to
        tempBlockmap2(i) = tBlockmap2; //row of element
        tempBlockmap3(i) = tBlockmap3; //column of element

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

umat amplitude::getraw_permuted(int u, int i, int n){
    //return index block, used for debugging and optimization
    umat aligned = fmBlocks(u)(i);
    if(n==0){
        aligned = aligned.rows(Pab(i));
    }
    if(n==1){
        aligned = aligned.rows(Pac(i));
    }
    if(n==2){
        aligned = aligned.rows(Pbc(i));
    }
    if(n==3){
        aligned = aligned.cols(Pij(i));
    }
    if(n==4){
        aligned = aligned.cols(Pik(i));
    }
    if(n==5){
        aligned = aligned.cols(Pjk(i));
    }
    return aligned;
}

mat amplitude::getblock_permuted(int u, int i, int n){
    //return index block, used for debugging and optimization

    umat aligned = fmBlocks(u)(i);
    if(n==0){
        aligned = aligned.rows(Pab(i));
    }
    if(n==1){
        aligned = aligned.rows(Pac(i));
    }
    if(n==2){
        aligned = aligned.rows(Pbc(i));
    }
    if(n==3){
        aligned = aligned.cols(Pij(i));
    }
    if(n==4){
        aligned = aligned.cols(Pik(i));
    }
    if(n==5){
        aligned = aligned.cols(Pjk(i));
    }
    if(n==6){
        aligned = aligned.rows(Pab(i));
        aligned = aligned.cols(Pij(i));
    }



    mat block = vElements.elem(aligned);
    block.reshape(fmBlocks(u)(i).n_rows, fmBlocks(u)(i).n_cols);

    return block;
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


// ###########################################
// ##
// ##  SEQUENCES OF ROWS/COLUMNS
// ##
// ############################################

field<ivec> amplitude::pp(){
    //return a "compacted" particle particle unique indexvector

    //length of "compacted" vector
    uint N = Np*(Np+1)/2;

    //indices
    ivec a(N);
    ivec b(N);
    uint count = 0;
    for(int na = 0; na<Np; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            a(count) = na;
            b(count) = nb;
            count += 1;
        }
    }

    ivec Kab = eBs.unique(conv_to<uvec>::from(a) +Nh) + eBs.unique(conv_to<uvec>::from(b)+Nh);

    field<ivec> ppmap(3);
    ppmap(0) = a;
    ppmap(1) = b;
    ppmap(2) = Kab;
    return ppmap;
    }

field<ivec> amplitude::hh_compact(){
    //return a "compacted" hole-hole unique indexvector

    //length of "compacted" vector
    uint N = Nh*(Nh+1)/2;

    //indices
    ivec a(N);
    ivec b(N);
    uint count = 0;
    for(int na = 0; na<Nh; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            a(count) = na;
            b(count) = nb;
            count += 1;
        }
    }

    ivec Kab = eBs.unique(conv_to<uvec>::from(a) ) + eBs.unique(conv_to<uvec>::from(b));

    field<ivec> ppmap(3);
    ppmap(0) = a;
    ppmap(1) = b;
    ppmap(2) = Kab;
    return ppmap;
}

field<ivec> amplitude::ppp(ivec signs){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Np*(Np+1)*(Np+2)/6;
    //indices
    ivec a(N);
    ivec b(N);
    ivec c(N);
    uint count = 0;
    for(int na = 0; na<Np; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            for(int nc = 0; nc<nb+1; ++nc){
                a(count) = na;
                b(count) = nb;
                c(count) = nc;
                count += 1;
            }
        }
    }
    ivec Kabc = signs(0)*eBs.unique(conv_to<uvec>::from(a) +Nh) + signs(1)*eBs.unique(conv_to<uvec>::from(b)+Nh) + signs(2)*eBs.unique(conv_to<uvec>::from(c) +Nh);

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = b;
    pppmap(2) = c;
    pppmap(3) = Kabc;
    return pppmap;
}

field<ivec> amplitude::hpp(){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Nh*Np*(Np+1)/2;
    //indices
    ivec i(N);
    ivec a(N);
    ivec b(N);
    uint count = 0;
    for(int ni = 0; ni<Nh; ++ni){
        for(int na = 0; na<Np; ++na){
            for(int nb = 0; nb<na+1; ++nb){
                i(count) = ni;
                a(count) = na;
                b(count) = nb;
                count += 1;
            }
        }
    }
    ivec Kiab = eBs.unique(conv_to<uvec>::from(i)) + eBs.unique(conv_to<uvec>::from(a)+Nh) + eBs.unique(conv_to<uvec>::from(b) +Nh);
    field<ivec> pppmap(4);
    pppmap(0) = i;
    pppmap(1) = a;
    pppmap(2) = b;
    pppmap(3) = Kiab;
    return pppmap;
}

field<ivec> amplitude::php(){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Nh*Np*(Np+1)/2;
    //indices
    ivec a(N);
    ivec i(N);
    ivec b(N);
    uint count = 0;
    for(int na = 0; na<Np; ++na){
        for(int ni = 0; ni<Nh; ++ni){
            for(int nb = 0; nb<na+1; ++nb){
                i(count) = ni;
                a(count) = na;
                b(count) = nb;
                count += 1;
            }
        }
    }
    ivec Kaib = eBs.unique(conv_to<uvec>::from(i)) + eBs.unique(conv_to<uvec>::from(a)+Nh) + eBs.unique(conv_to<uvec>::from(b) +Nh);

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = i;
    pppmap(2) = b;
    pppmap(3) = Kaib;
    return pppmap;
}

field<ivec> amplitude::pph(){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Nh*Np*(Np+1)/2;
    //indices
    ivec i(N);
    ivec a(N);
    ivec b(N);
    uint count = 0;
    for(int na = 0; na<Np; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            for(int ni = 0; ni<Nh; ++ni){
                i(count) = ni;
                a(count) = na;
                b(count) = nb;
                count += 1;
            }
        }
    }
    ivec Kabi = eBs.unique(conv_to<uvec>::from(i)) + eBs.unique(conv_to<uvec>::from(a)+Nh) + eBs.unique(conv_to<uvec>::from(b) +Nh);

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = b;
    pppmap(2) = i;
    pppmap(3) = Kabi;
    return pppmap;
}

field<ivec> amplitude::hh(){
    //noncompacted
    ivec h = linspace<ivec>(0,Nh-1,Nh);
    field<ivec> hhmap(3);
    hhmap(0) = kron(ones<ivec>(Nh), h);
    hhmap(1) = kron(h, ones<ivec>(Nh));
    hhmap(2) = eBs.unique(conv_to<uvec>::from(hhmap(0))) + eBs.unique(conv_to<uvec>::from(hhmap(1)));
    return hhmap;
}

/*
field<ivec> amplitude::hhh(){
    //noncompacted
    ivec h = linspace<ivec>(0,Nh*Nh*Nh-1,Nh*Nh*Nh);
    field<ivec> hhhmap(4);

    hhhmap(2) = conv_to<ivec>::from(floor(h/(Nh*Nh))); //k
    hhhmap(1) = conv_to<ivec>::from(floor(((h - hhhmap(2)*Nh*Nh))/Nh));
    hhhmap(0) = conv_to<ivec>::from(h - hhhmap(2)*Nh*Nh - hhhmap(1)*Nh);

    hhhmap(3) = eBs.unique(conv_to<uvec>::from(hhhmap(0))) + eBs.unique(conv_to<uvec>::from(hhhmap(1)))+eBs.unique(conv_to<uvec>::from(hhhmap(2)));
    return hhhmap;
}*/

field<ivec> amplitude::hhh(){
    //return a "compacted" hole-hole-hole particle unique indexvector
    //length of "compacted" vector
    uint N = Nh*(Nh+1)*(Nh+2)/6;
    //indices
    ivec a(N);
    ivec b(N);
    ivec c(N);
    uint count = 0;
    for(int na = 0; na<Nh; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            for(int nc = 0; nc<nb+1; ++nc){
                a(count) = na;
                b(count) = nb;
                c(count) = nc;
                count += 1;
            }
        }
    }
    ivec Kabc = eBs.unique(conv_to<uvec>::from(a)) + eBs.unique(conv_to<uvec>::from(b)) + eBs.unique(conv_to<uvec>::from(c));

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = b;
    pppmap(2) = c;
    pppmap(3) = Kabc;
    return pppmap;
}

field<ivec> amplitude::hp(){
    ivec h = linspace<ivec>(0,Nh-1,Nh);
    ivec p = linspace<ivec>(0,Np-1,Np);
    field<ivec> hhmap(3);
    hhmap(0) = kron(h, ones<ivec>(Np));
    hhmap(1) = kron(ones<ivec>(Nh), p);
    hhmap(2) = eBs.unique(conv_to<uvec>::from(hhmap(0))) + eBs.unique(conv_to<uvec>::from(hhmap(1))+Nh);
    return hhmap;
}

field<ivec> amplitude::ph(){
    ivec h = linspace<ivec>(0,Nh-1,Nh);
    ivec p = linspace<ivec>(0,Np-1,Np);
    field<ivec> hhmap(3);
    hhmap(0) = kron(p, ones<ivec>(Nh));
    hhmap(1) = kron(ones<ivec>(Np), h);
    hhmap(2) = eBs.unique(conv_to<uvec>::from(hhmap(0))+Nh) + eBs.unique(conv_to<uvec>::from(hhmap(1)));
    return hhmap;
}

field<uvec> amplitude::partition(ivec LHS, ivec K_unique){
    //identify i blocks of LHS where LHS==K(i)
    //Note that by "blocks", we actually means indices corresponding to preserved quantum numbers
    uvec l_sorted = sort_index(LHS);
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);
    uvec row(1000000); //arbitrarily large number (biggest possible block)
    uint nx = 0;
    int l_c= LHS(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool row_collect = false;
    //int ll_c = l_c;

    while(lc < uiS){
        l_c = LHS(l_sorted(lc));

        //if element belongs to block
        if(l_c == C){
            //collect element
            row(nx) = l_sorted(lc);
            nx += 1;
            row_collect = true;
        }
        //if row is complete
        else{
            //if row contains elements
            if(row_collect){
                //collect block
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


    return tempRows;

}


field<uvec> amplitude::partition_pp(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(2));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(2)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    Pab.set_size(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(2)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint a;
    uint b;
    //uint c;
    uint l_sorted_lc;
    uvec row(1000000);
    //uvec row_b(1000000);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec pab(1000000);

    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(2)(l_sorted_lc);

        if(l_c == C){
            a = Na(l_sorted_lc);
            b = Nb(l_sorted_lc);
            //c = Cn(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block


            //row_b(nx) = b;
            if(a!=b){
                row(nx) = a + b*Np;
                row(nx+1) = b + a*Np;
                pab(nx) = nx+1;
                pab(nx+1) = nx;


                nx += 2;

                //row_b(nx) = a;
            }
            else{
                row(nx) = a + b*Np;
                pab(nx) = nx;
                nx += 1;
            }
            //row(nx) = l_sorted(lc);
            //nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                uvec rr = sort(row(span(0,nx-1)));
                //uvec rr = sort_index(row);
                uvec row_rr = row(span(0,nx-1));
                uvec ab_index = pab(span(0,nx-1));
                tempRows(i) = row_rr.elem(rr);
                //tempRows(i) = row.elem(rr);
                Pab(i) = ab_index.elem(rr.elem(ab_index));
                //tempRows(1)(i) = row_b(span(0,nx-1));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    tempRows(i) = sort(row(span(0,nx-1)));
    return tempRows;
}

field<uvec> amplitude::partition_ppp(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint p,q,r;

    uint l_sorted_lc;
    uvec row(1000000);
    //uvec row_b(1000000);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec Nc = conv_to<uvec>::from(LHS(2));

    uint Np2 =Np*Np;
    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            p = Na(l_sorted_lc);
            q = Nb(l_sorted_lc);
            r = Nc(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block
            //a b c
            row(nx) = p + q*Np + r*Np2;
            nx += 1;
            if(r!=q){
                //b!=c
                row(nx) = p + r*Np + q*Np2;
                nx += 1;
                if(p!=q){
                    //a!=b (!=c)
                    row(nx) = q + p*Np + r*Np2;
                    nx += 1;
                    row(nx) = q + r*Np + p*Np2;
                    nx += 1;
                    //it follows that p!=r, so
                    row(nx) = r + p*Np + q*Np2;
                    nx += 1;
                    row(nx) = r + q*Np + p*Np2;
                    nx += 1;
                }
            }
            else{
                if(p!=q){
                    row(nx) = q + p*Np + r*Np2;
                    nx += 1;
                }
                if(p!=r){
                    row(nx) = r + q*Np + p*Np2;
                    nx += 1;
                }
            }
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
    }

    //collect final block
    //tempRows(i) = sort(row(span(0,nx-1))); //is this needed?
    return tempRows;
}

field<uvec> amplitude::partition_hh_permutations(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(2));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(2)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    Pij.set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(2)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool row_collect = false;
    uint a;
    uint b;
    uint l_sorted_lc;
    uvec row(1000000);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec pab(1000000);

    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(2)(l_sorted_lc);

        if(l_c == C){
            a = Na(l_sorted_lc);
            b = Nb(l_sorted_lc);
            //c = Cn(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block


            //row_b(nx) = b;
            if(a!=b){
                row(nx) = a + b*Nh;
                row(nx+1) = b + a*Nh;
                pab(nx) = nx+1;
                pab(nx+1) = nx;


                nx += 2;

                //row_b(nx) = a;
            }
            else{
                row(nx) = a + b*Nh;
                pab(nx) = nx;
                nx += 1;
            }
            //row(nx) = l_sorted(lc);
            //nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                uvec rr = sort_index(row(span(0,nx-1)));
                uvec r_inv = sort_index(rr);

                //uvec rr = sort_index(row);
                uvec row_rr = row(span(0,nx-1));
                uvec ab_index = pab(span(0,nx-1));
                tempRows(i) = row_rr.elem(rr);
                //tempRows(i) = row.elem(rr);
                uvec ab_perm(ab_index.n_rows);
                for(uint h =0; h < ab_index.n_rows; ++h){
                    uint a_new = r_inv(h);
                    uint b_new = r_inv(ab_index(h));
                    ab_perm(a_new) = b_new;
                    ab_perm(b_new) = a_new;
                }


                //Pij(i) = ab_index.elem(rr);
                Pij(i) = ab_perm;
                //tempRows(1)(i) = row_b(span(0,nx-1));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    tempRows(i) = sort(row(span(0,nx-1)));
    return tempRows;

}

field<uvec> amplitude::partition_pp_permutations(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(2));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(2)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    Pab.set_size(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(2)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint a;
    uint b;
    //uint c;
    uint l_sorted_lc;
    uvec row(1000000);
    //uvec row_b(1000000);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec pab(1000000);

    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(2)(l_sorted_lc);

        if(l_c == C){
            a = Na(l_sorted_lc);
            b = Nb(l_sorted_lc);
            //c = Cn(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block


            //row_b(nx) = b;
            if(a!=b){
                row(nx) = a + b*Np;
                row(nx+1) = b + a*Np;
                pab(nx) = nx+1;
                pab(nx+1) = nx;


                nx += 2;

                //row_b(nx) = a;
            }
            else{
                row(nx) = a + b*Np;
                pab(nx) = nx;
                nx += 1;
            }
            //row(nx) = l_sorted(lc);
            //nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                uvec rr = sort_index(row(span(0,nx-1)));
                uvec r_inv = sort_index(rr);
                //uvec arr = sort_index(rr);
                uvec row_rr = row(span(0,nx-1));
                uvec ab_index = pab(span(0,nx-1));
                //rr.print();
                //cout << endl;
                //row_rr.print();

                //uvec te = row_rr.elem(rr);
                tempRows(i) = row_rr.elem(rr);
                //tempRows(i) = row.elem(rr);
                uvec ab_perm(ab_index.n_rows);
                for(uint h =0; h < ab_index.n_rows; ++h){
                    uint a_new = r_inv(h);
                    uint b_new = r_inv(ab_index(h));
                    ab_perm(a_new) = b_new;
                    ab_perm(b_new) = a_new;
                }

                //Pab(i) = rr.elem(arr.elem(ab_index));
                Pab(i) = ab_perm;
                //tempRows(1)(i) = row_b(span(0,nx-1));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    //tempRows(i) = sort(row(span(0,nx-1)));
    return tempRows;

}


field<uvec> amplitude::partition_ppp_permutations(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    Pab.set_size(uiN);
    Pac.set_size(uiN);
    Pbc.set_size(uiN);

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint a,b,c;

    uint l_sorted_lc;
    uvec row(1000000);
    //uvec row_b(1000000);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec Nc = conv_to<uvec>::from(LHS(2));

    uvec permute_ab(1000000);
    uvec permute_ac(1000000);
    uvec permute_bc(1000000);



    uint nx0 = 0;
    uint Np2 =Np*Np;
    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            a = Na(l_sorted_lc);
            b = Nb(l_sorted_lc);
            c = Nc(l_sorted_lc);
            if(a==b){
                if(a==c){
                    //(2)
                    row(nx) = a + b*Np + c*Np2;

                    permute_ab(nx) = nx;

                    permute_ac(nx) = nx;

                    permute_bc(nx) = nx;

                    nx += 1;

                }
                else{
                    //(3)
                    row(nx) = a + b*Np + c*Np2;
                    row(nx+1) = a + c*Np + b*Np2;
                    row(nx+2) = c + b*Np + a*Np2;

                    permute_ab(nx) = nx;
                    permute_ab(nx+1) = nx +2;
                    permute_ab(nx+2) = nx+1;

                    permute_ac(nx) = nx+2;
                    permute_ac(nx+1) = nx+1;
                    permute_ac(nx+2) = nx;

                    permute_bc(nx) = nx+1;
                    permute_bc(nx+1) = nx;
                    permute_bc(nx+2) = nx+2;



                    nx += 3;

                }
            }
            else{
                //a!=b
                if(a==c){
                    //(1)
                    row(nx) = a + b*Np + c*Np2;
                    row(nx+1) = b + a*Np + c*Np2;
                    row(nx+2) = a + c*Np + b*Np2;

                    permute_ab(nx) = nx+1;
                    permute_ab(nx+1) = nx;
                    permute_ab(nx+2) = nx+2;

                    permute_ac(nx) = nx;
                    permute_ac(nx+1) = nx+2;
                    permute_ac(nx+2) = nx+1;

                    permute_bc(nx) = nx+2;
                    permute_bc(nx+1) = nx+1;
                    permute_bc(nx+2) = nx;

                    nx += 3;



                }
                else{
                    if(b==c){
                        //(4)
                        row(nx) = a + b*Np + c*Np2;
                        row(nx+1) = b + a*Np + c*Np2;
                        row(nx+2) = c + b*Np + a*Np2;

                        permute_ab(nx) = nx+1;
                        permute_ab(nx+1) = nx;
                        permute_ab(nx+2) = nx+2;

                        permute_ac(nx) = nx+2;
                        permute_ac(nx+1) = nx+1;
                        permute_ac(nx+2) = nx;

                        permute_bc(nx) = nx;
                        permute_bc(nx+1) = nx+2;
                        permute_bc(nx+2) = nx+1;

                        nx += 3;

                    }
                    else{
                        //(5)
                        row(nx) =   a + b*Np + c*Np2;
                        row(nx+1) = b + a*Np + c*Np2;
                        row(nx+2) = a + c*Np + b*Np2;
                        row(nx+3) = c + a*Np + b*Np2;
                        row(nx+4) = b + c*Np + a*Np2;
                        row(nx+5) = c + b*Np + a*Np2;

                        permute_ab(nx) = nx+1;
                        permute_ab(nx+1) = nx;
                        permute_ab(nx+2) = nx+3;
                        permute_ab(nx+3) = nx+2;
                        permute_ab(nx+4) = nx+5;
                        permute_ab(nx+5) = nx+4;

                        permute_ac(nx) = nx+5;
                        permute_ac(nx+1) = nx+3;
                        permute_ac(nx+2) = nx+4;
                        permute_ac(nx+3) = nx+5;
                        permute_ac(nx+4) = nx+2;
                        permute_ac(nx+5) = nx;

                        permute_bc(nx) = nx+2;
                        permute_bc(nx+1) = nx+4;
                        permute_bc(nx+2) = nx;
                        permute_bc(nx+3) = nx+5;
                        permute_bc(nx+4) = nx+1;
                        permute_bc(nx+5) = nx+3;


                        nx += 6;

                    }
                }
            }


            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                uvec sorted = sort_index(row(span(0,nx-1)));
                uvec rr = row(span(0,nx-1));
                uvec pab = permute_ab(span(0,nx-1));
                uvec pac = permute_ac(span(0,nx-1));
                uvec pbc = permute_bc(span(0,nx-1));

                tempRows(i) = rr.elem(sorted);




                uvec r_inv = sort_index(sorted);
                uvec ab_perm(pab.n_rows);
                uvec ac_perm(pab.n_rows);
                uvec bc_perm(pab.n_rows);


                for(uint h =0; h < ab_perm.n_rows; ++h){
                    uint current = r_inv(h);

                    uint ab_new = r_inv(pab(h));
                    ab_perm(current) = ab_new;
                    ab_perm(ab_new) = current;

                    uint ac_new = r_inv(pac(h));
                    ac_perm(current) = ac_new;
                    ac_perm(ac_new) = current;

                    uint bc_new = r_inv(pbc(h));
                    bc_perm(current) = bc_new;
                    bc_perm(ac_new) = current;


                }

                Pab(i) = ab_perm; //pab.elem(sorted);
                Pac(i) = ac_perm; //pac.elem(sorted);
                Pbc(i) = bc_perm; //pbc.elem(sorted);

                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    //tempRows(i) = sort(row(span(0,nx-1))); //is this needed?

    uvec sorted = sort_index(row(span(0,nx-1)));
    uvec rr = row(span(0,nx-1));
    uvec pab = permute_ab(span(0,nx-1));
    uvec pac = permute_ac(span(0,nx-1));
    uvec pbc = permute_bc(span(0,nx-1));

    tempRows(i) = rr.elem(sorted);




    uvec r_inv = sort_index(sorted);
    uvec ab_perm(pab.n_rows);
    uvec ac_perm(pab.n_rows);
    uvec bc_perm(pab.n_rows);


    for(uint h =0; h < ab_perm.n_rows; ++h){
        uint current = r_inv(h);

        uint ab_new = r_inv(pab(h));
        ab_perm(current) = ab_new;
        ab_perm(ab_new) = current;

        uint ac_new = r_inv(pac(h));
        ac_perm(current) = ac_new;
        ac_perm(ac_new) = current;

        uint bc_new = r_inv(pbc(h));
        bc_perm(current) = bc_new;
        bc_perm(ac_new) = current;


    }

    Pab(i) = ab_perm; //pab.elem(sorted);
    Pac(i) = ac_perm; //pac.elem(sorted);
    Pbc(i) = bc_perm; //pbc.elem(sorted);


    return tempRows;
}

field<uvec> amplitude::partition_hhh_permutations(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    Pij.set_size(uiN);
    Pik.set_size(uiN);
    Pjk.set_size(uiN);

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint a,b,c;

    uint l_sorted_lc;
    uvec row(1000000);
    //uvec row_b(1000000);


    uvec Ni = conv_to<uvec>::from(LHS(0));
    uvec Nj = conv_to<uvec>::from(LHS(1));
    uvec Nk = conv_to<uvec>::from(LHS(2));

    uvec permute_ij(1000000);
    uvec permute_ik(1000000);
    uvec permute_jk(1000000);



    uint nx0 = 0;
    uint Nh2 =Nh*Nh;
    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            a = Ni(l_sorted_lc);
            b = Nj(l_sorted_lc);
            c = Nk(l_sorted_lc);
            if(a==b){
                if(a==c){
                    //(2)
                    row(nx) = a + b*Nh + c*Nh2;

                    permute_ij(nx) = nx;

                    permute_ik(nx) = nx;

                    permute_jk(nx) = nx;

                    nx += 1;

                }
                else{
                    //(3)
                    row(nx) = a + b*Nh + c*Nh2;
                    row(nx+1) = a + c*Nh + b*Nh2;
                    row(nx+2) = c + b*Nh + a*Nh2;

                    permute_ij(nx) = nx;
                    permute_ij(nx+1) = nx +2;
                    permute_ij(nx+2) = nx+1;

                    permute_ik(nx) = nx+2;
                    permute_ik(nx+1) = nx+1;
                    permute_ik(nx+2) = nx;

                    permute_jk(nx) = nx+1;
                    permute_jk(nx+1) = nx;
                    permute_jk(nx+2) = nx+2;



                    nx += 3;

                }
            }
            else{
                //a!=b
                if(a==c){
                    //(1)
                    row(nx) = a + b*Nh + c*Nh2;
                    row(nx+1) = b + a*Nh + c*Nh2;
                    row(nx+2) = a + c*Nh + b*Nh2;

                    permute_ij(nx) = nx+1;
                    permute_ij(nx+1) = nx;
                    permute_ij(nx+2) = nx+2;

                    permute_ik(nx) = nx;
                    permute_ik(nx+1) = nx+2;
                    permute_ik(nx+2) = nx+1;

                    permute_jk(nx) = nx+2;
                    permute_jk(nx+1) = nx+1;
                    permute_jk(nx+2) = nx;

                    nx += 3;



                }
                else{
                    if(b==c){
                        //(4)
                        row(nx) = a + b*Nh + c*Nh2;
                        row(nx+1) = b + a*Nh + c*Nh2;
                        row(nx+2) = c + b*Nh + a*Nh2;

                        permute_ij(nx) = nx+1;
                        permute_ij(nx+1) = nx;
                        permute_ij(nx+2) = nx+2;

                        permute_ik(nx) = nx+2;
                        permute_ik(nx+1) = nx+1;
                        permute_ik(nx+2) = nx;

                        permute_jk(nx) = nx;
                        permute_jk(nx+1) = nx+2;
                        permute_jk(nx+2) = nx+1;

                        nx += 3;

                    }
                    else{
                        //(5)
                        row(nx) =   a + b*Nh + c*Nh2;
                        row(nx+1) = b + a*Nh + c*Nh2;
                        row(nx+2) = a + c*Nh + b*Nh2;
                        row(nx+3) = c + a*Nh + b*Nh2;
                        row(nx+4) = b + c*Nh + a*Nh2;
                        row(nx+5) = c + b*Nh + a*Nh2;

                        permute_ij(nx) = nx+1;
                        permute_ij(nx+1) = nx;
                        permute_ij(nx+2) = nx+3;
                        permute_ij(nx+3) = nx+2;
                        permute_ij(nx+4) = nx+5;
                        permute_ij(nx+5) = nx+4;

                        permute_ik(nx) = nx+5;
                        permute_ik(nx+1) = nx+3;
                        permute_ik(nx+2) = nx+4;
                        permute_ik(nx+3) = nx+5;
                        permute_ik(nx+4) = nx+2;
                        permute_ik(nx+5) = nx;

                        permute_jk(nx) = nx+2;
                        permute_jk(nx+1) = nx+4;
                        permute_jk(nx+2) = nx;
                        permute_jk(nx+3) = nx+5;
                        permute_jk(nx+4) = nx+1;
                        permute_jk(nx+5) = nx+3;


                        nx += 6;

                    }
                }
            }


            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                uvec sorted = sort_index(row(span(0,nx-1)));
                uvec rr = row(span(0,nx-1));
                uvec pij = permute_ij(span(0,nx-1));
                uvec pik = permute_ik(span(0,nx-1));
                uvec pjk = permute_jk(span(0,nx-1));

                tempRows(i) = rr.elem(sorted);



                uvec r_inv = sort_index(sorted);
                uvec ab_perm(pij.n_rows);
                uvec ac_perm(pik.n_rows);
                uvec bc_perm(pjk.n_rows);


                for(uint h =0; h < ab_perm.n_rows; ++h){
                    uint current = r_inv(h);

                    uint ab_new = r_inv(pij(h));
                    ab_perm(current) = ab_new;
                    ab_perm(ab_new) = current;

                    uint ac_new = r_inv(pik(h));
                    ac_perm(current) = ac_new;
                    ac_perm(ac_new) = current;

                    uint bc_new = r_inv(pjk(h));
                    bc_perm(current) = bc_new;
                    bc_perm(ac_new) = current;


                }

                Pij(i) = ab_perm; //pab.elem(sorted);
                Pik(i) = ac_perm; //pac.elem(sorted);
                Pjk(i) = bc_perm; //pbc.elem(sorted);


                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    //tempRows(i) = sort(row(span(0,nx-1))); //is this needed?
    uvec sorted = sort_index(row(span(0,nx-1)));
    uvec rr = row(span(0,nx-1));
    uvec pij = permute_ij(span(0,nx-1));
    uvec pik = permute_ik(span(0,nx-1));
    uvec pjk = permute_jk(span(0,nx-1));

    tempRows(i) = rr.elem(sorted);



    uvec r_inv = sort_index(sorted);
    uvec ab_perm(pij.n_rows);
    uvec ac_perm(pik.n_rows);
    uvec bc_perm(pjk.n_rows);


    for(uint h =0; h < ab_perm.n_rows; ++h){
        uint current = r_inv(h);

        uint ab_new = r_inv(pij(h));
        ab_perm(current) = ab_new;
        ab_perm(ab_new) = current;

        uint ac_new = r_inv(pik(h));
        ac_perm(current) = ac_new;
        ac_perm(ac_new) = current;

        uint bc_new = r_inv(pjk(h));
        bc_perm(current) = bc_new;
        bc_perm(ac_new) = current;


    }

    Pij(i) = ab_perm; //pab.elem(sorted);
    Pik(i) = ac_perm; //pac.elem(sorted);
    Pjk(i) = bc_perm; //pbc.elem(sorted);

    return tempRows;
}

field<uvec> amplitude::partition_hpp(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    //bool adv = false;

    int Nhp = Np*Nh;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint p,q,r;

    uint l_sorted_lc;
    uvec row(1000000);
    //uvec row_b(1000000);


    uvec Ni = conv_to<uvec>::from(LHS(0));
    uvec Na = conv_to<uvec>::from(LHS(1));
    uvec Nb = conv_to<uvec>::from(LHS(2));

    uint Np2 =Np*Np;
    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            p = Ni(l_sorted_lc);
            q = Na(l_sorted_lc);
            r = Nb(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block
            row(nx) = p + q*Nh + r*Nhp;
            nx += 1;
            if(r!=q){
                row(nx) = p + r*Nh + q*Nhp;
                nx += 1;
            }
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
    }

    //collect final block
    tempRows(i) = sort(row(span(0,nx-1)));
    return tempRows;
}
