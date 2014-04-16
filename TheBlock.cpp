#include "FreeFunctions.h"
#include "ESolver.h"

using namespace Eigen;

TheBlock::TheBlock(int m, const std::vector<int>& qNumList, const MatrixXd& hS,
                   const std::vector<MatrixXd>& off0RhoBasisH2,
                   const std::vector<MatrixXd>& off1RhoBasisH2, int l)
    : m(m), qNumList(qNumList), hS(hS), off0RhoBasisH2(off0RhoBasisH2),
      off1RhoBasisH2(off1RhoBasisH2), l(l) {};

TheBlock::TheBlock(const Hamiltonian& ham)
    : m(d), qNumList(ham.oneSiteQNums), hS(MatrixDd::Zero()), l(0)
{
    off0RhoBasisH2.assign(ham.h2.begin(),
                          ham.h2.begin() + indepCouplingOperators);
};

TheBlock TheBlock::nextBlock(const stepData& data, rmMatrixXd& psiGround)
{
    std::vector<int> hSprimeQNumList      // add in quantum numbers of new site
        = vectorProductSum(qNumList, data.ham.oneSiteQNums);
    MatrixXd hSprime = kp(hS, Id_d)
                       + data.ham.blockAdjacentSiteJoin(1, off0RhoBasisH2);
                                                       // expanded system block
    
//    if(l != 0)
//        hSprime += data.ham.blockAdjacentSiteJoin(2, off1RhoBasisH2);
    
    hSprime += (l == 0 ? -data.ham.blockAdjacentSiteJoin(1, off0RhoBasisH2) / 2
                       : data.ham.blockAdjacentSiteJoin(2, off1RhoBasisH2));
    std::vector<MatrixXd> tempOff0RhoBasisH2,
                          tempOff1RhoBasisH2;
    tempOff0RhoBasisH2.reserve(indepCouplingOperators);
    tempOff1RhoBasisH2.reserve(indepCouplingOperators);
    int md = m * d;
    if(data.exactDiag)
    { // if near edge of system, no truncation necessary so skip DMRG algorithm
        for(int i = 0; i < indepCouplingOperators; i++)
        {
            tempOff0RhoBasisH2.push_back(kp(Id(m), data.ham.h2[i]));
            tempOff1RhoBasisH2.push_back(kp(off0RhoBasisH2[i], Id_d));
        };
        return TheBlock(md, hSprimeQNumList, hSprime, tempOff0RhoBasisH2,
                        tempOff1RhoBasisH2, l + 1);
    };
    int compm = data.compBlock -> m,
        compmd = compm * d;
    HamSolver hSuperSolver = (data.infiniteStage ? // find superblock eigenstates
                              HamSolver(MatrixXd(kp(hSprime, Id(md))
                                                 + data.ham.lBlockrSiteJoin(off0RhoBasisH2, m)
                                                 + data.ham.siteSiteJoin(m, m)
                                                 + data.ham.lSiterBlockJoin(m, off0RhoBasisH2)
                                                 + kp(Id(md), hSprime)),
                                        vectorProductSum(hSprimeQNumList,
                                                         hSprimeQNumList),
                                        data.ham.targetQNum * (l + 2) / data.ham.lSys * 2,
                                        psiGround, data.lancTolerance) :
                                               // int automatically rounds down
                              HamSolver(MatrixXd(kp(hSprime, Id(compmd))
                                                 + data.ham.lBlockrSiteJoin(off0RhoBasisH2, compm)
                                                 + data.ham.siteSiteJoin(m, compm)
                                                 + data.ham.lSiterBlockJoin(m, data.compBlock -> off0RhoBasisH2)
                                                 + kp(Id(md), data.ham.blockAdjacentSiteJoin(1, data.compBlock -> off0RhoBasisH2)
                                                              + data.ham.blockAdjacentSiteJoin(2, data.compBlock -> off1RhoBasisH2)
                                                              + kp(data.compBlock -> hS, Id_d))),
                                        vectorProductSum(hSprimeQNumList,
                                                         vectorProductSum(data.compBlock -> qNumList,
                                                                          data.ham.oneSiteQNums)),
                                        data.ham.targetQNum, psiGround,
                                        data.lancTolerance));
    psiGround = hSuperSolver.lowestEvec;                        // ground state
    psiGround.resize(md, compmd);
    DMSolver rhoSolver(psiGround * psiGround.adjoint(), hSprimeQNumList,
                       data.mMax);           // find density matrix eigenstates
    primeToRhoBasis = rhoSolver.highestEvecs; // construct change-of-basis matrix
    for(int i = 0; i < indepCouplingOperators; i++)
    {
        tempOff0RhoBasisH2.push_back(changeBasis(kp(Id(m), data.ham.h2[i])));
        tempOff1RhoBasisH2.push_back(changeBasis(kp(off0RhoBasisH2[i], Id_d)));
    };
    if(!data.infiniteStage) // modify psiGround to predict the next ground state
    {
        for(int sPrimeIndex = 0; sPrimeIndex < md; sPrimeIndex++)
                    // transpose the environment block and right-hand free site
        {
            rmMatrixXd ePrime = psiGround.row(sPrimeIndex);
            ePrime.resize(compm, d);
            ePrime.transposeInPlace();
            ePrime.resize(1, compmd);
            psiGround.row(sPrimeIndex) = ePrime;
        };
        psiGround = primeToRhoBasis.adjoint() * psiGround; 
                                      // change the expanded system block basis
        psiGround.resize(data.mMax * d, compm);
        psiGround *= data.beforeCompBlock -> primeToRhoBasis.transpose();
                                          // change the environment block basis
        psiGround.resize(data.mMax * d
                         * data.beforeCompBlock -> primeToRhoBasis.rows(), 1);
    };
    return TheBlock(data.mMax, rhoSolver.highestEvecQNums, changeBasis(hSprime),
                    tempOff0RhoBasisH2, tempOff1RhoBasisH2, l + 1);
                                  // save expanded-block operators in new basis
};

EffectiveHamiltonian TheBlock::createHSuperFinal(const stepData& data,
                                                 const rmMatrixXd& psiGround,
                                                 int skips) const
{
    int compm = data.compBlock -> m;
    return EffectiveHamiltonian(MatrixXd(kp(kp(hS, Id_d)
                                            + data.ham.blockAdjacentSiteJoin(1, off0RhoBasisH2)
                                            + data.ham.blockAdjacentSiteJoin(2, off1RhoBasisH2),
                                            Id(compm * d))
                                         + data.ham.lBlockrSiteJoin(off0RhoBasisH2, compm)
                                         + data.ham.siteSiteJoin(m, compm)
                                         + data.ham.lSiterBlockJoin(m, data.compBlock -> off0RhoBasisH2)
                                         + kp(Id(m * d), data.ham.blockAdjacentSiteJoin(1, data.compBlock -> off0RhoBasisH2)
                                                         + data.ham.blockAdjacentSiteJoin(2, data.compBlock -> off1RhoBasisH2)
                                                         + kp(data.compBlock -> hS, Id_d))),
                                qNumList, data.compBlock -> qNumList, data,
                                psiGround, m, compm, skips);
};

MatrixXd TheBlock::changeBasis(const MatrixXd& mat) const
{
	return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};
