#include "FreeFunctions.h"
#include "ESolver.h"

using namespace Eigen;

Hamiltonian TheBlock::ham;
rmMatrixXd TheBlock::psiGround;
int TheBlock::mMax;
bool TheBlock::firstfDMRGStep;

TheBlock::TheBlock(int m, const MatrixXd& hS,
                   const std::vector<MatrixXd>& off0RhoBasisH2,
                   const std::vector<MatrixXd>& off1RhoBasisH2,
                   const std::vector<int>& qNumList)
    : qNumList(qNumList), hS(hS), off0RhoBasisH2(off0RhoBasisH2),
      off1RhoBasisH2(off1RhoBasisH2), m(m) {};

TheBlock::TheBlock(const Hamiltonian& hamIn, int mMaxIn)
    : qNumList(ham.oneSiteQNums), hS(MatrixDd::Zero()), m(d)
{
    firstfDMRGStep = true;
    ham = hamIn;
    mMax = mMaxIn;
    off0RhoBasisH2.assign(ham.h2.begin(),
                          ham.h2.begin() + indepCouplingOperators);
};

TheBlock TheBlock::nextBlock(int l, TheBlock& compBlock, bool exactDiag,
                             bool infiniteStage,
                             const TheBlock& beforeCompBlock)
{
    std::vector<int> hSprimeQNumList	// add in quantum numbers of new site
        = vectorProductSum(qNumList, ham.oneSiteQNums);
    MatrixXd hSprime = kp(hS, Id_d) + ham.blockAdjacentSiteJoin(1,
                                                                off0RhoBasisH2);
                                                       // expanded system block
    
//    if(l != 0)
//        hSprime += ham.blockAdjacentSiteJoin(2, off1RhoBasisH2);
    
    hSprime += (l == 0 ? -ham.blockAdjacentSiteJoin(1, off0RhoBasisH2) / 2
                       : ham.blockAdjacentSiteJoin(2, off1RhoBasisH2));
    
    std::vector<MatrixXd> tempOff0RhoBasisH2,
                          tempOff1RhoBasisH2;
    tempOff0RhoBasisH2.reserve(indepCouplingOperators);
    tempOff1RhoBasisH2.reserve(indepCouplingOperators);
    int md = m * d;
    if(exactDiag)
    { // if near edge of system, no truncation necessary so skip DMRG algorithm
        for(int i = 0; i < indepCouplingOperators; i++)
        {
            tempOff0RhoBasisH2.push_back(kp(Id(m), ham.h2[i]));
            tempOff1RhoBasisH2.push_back(kp(off0RhoBasisH2[i], Id_d));
        };
        return TheBlock(md, hSprime, tempOff0RhoBasisH2, tempOff1RhoBasisH2,
                        hSprimeQNumList);
    };
    VectorXd seed;
    int compm = compBlock.m,
        compmd = compm * d;
    if(infiniteStage)
    {
        seed = VectorXd::Random(md * md);
        seed /= seed.norm();
    }
    else if(firstfDMRGStep)
    {
        seed = VectorXd::Random(md * compmd);
        seed /= seed.norm();
        firstfDMRGStep = false;
    }
    else
        seed = psiGround;
    HamSolver hSuperSolver = (infiniteStage ?       // find superblock eigenstates
                              HamSolver(MatrixXd(kp(hSprime, Id(md))
                                                 + ham.lBlockrSiteJoin(off0RhoBasisH2, m)
                                                 + ham.siteSiteJoin(m, m)
                                                 + ham.lSiterBlockJoin(m, off0RhoBasisH2)
                                                 + kp(Id(md), hSprime)),
                                        vectorProductSum(hSprimeQNumList,
                                                         hSprimeQNumList),
                                        ham.targetQNum * (l + 2) / ham.lSys * 2,
                                        seed) : // int automatically rounds down
                              HamSolver(MatrixXd(kp(hSprime, Id(compmd))
                                                 + ham.lBlockrSiteJoin(off0RhoBasisH2, compm)
                                                 + ham.siteSiteJoin(m, compm)
                                                 + ham.lSiterBlockJoin(m, compBlock.off0RhoBasisH2)
                                                 + kp(Id(md), ham.blockAdjacentSiteJoin(1, compBlock.off0RhoBasisH2)
                                                              + ham.blockAdjacentSiteJoin(2, compBlock.off1RhoBasisH2)
                                                              + kp(compBlock.hS, Id_d))),
                                        vectorProductSum(hSprimeQNumList,
                                                         vectorProductSum(compBlock.qNumList,
                                                                          ham.oneSiteQNums)),
                                        ham.targetQNum, seed));
    psiGround = hSuperSolver.lowestEvec;                        // ground state
    psiGround.resize(md, infiniteStage ? md : compmd);
    DMSolver rhoSolver(psiGround * psiGround.adjoint(), hSprimeQNumList, mMax);
                                            // find density matrix eigenstates
    primeToRhoBasis = rhoSolver.highestEvecs; // construct change-of-basis matrix
    for(int i = 0; i < indepCouplingOperators; i++)
    {
        tempOff0RhoBasisH2.push_back(changeBasis(kp(Id(m), ham.h2[i])));
        tempOff1RhoBasisH2.push_back(changeBasis(kp(off0RhoBasisH2[i], Id_d)));
    };
    if(infiniteStage)                // copy primeToRhoBasis to reflected block
        compBlock.primeToRhoBasis = primeToRhoBasis;
    else                   // modify psiGround to predict the next ground state
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
        psiGround.resize(mMax * d, compm);
        psiGround *= beforeCompBlock.primeToRhoBasis.transpose();
                                          // change the environment block basis
        psiGround.resize(mMax * d * beforeCompBlock.primeToRhoBasis.rows(), 1);
    };
    return TheBlock(mMax, changeBasis(hSprime), tempOff0RhoBasisH2,
                    tempOff1RhoBasisH2, rhoSolver.highestEvecQNums);
                                // save expanded-block operators in new basis
};

void TheBlock::randomSeed(const TheBlock& compBlock)
{
    psiGround = VectorXd::Random(m * d * compBlock.m * d);
    psiGround /= psiGround.norm();
};

void TheBlock::reflectPredictedPsi()
{
    psiGround.resize(mMax * d, m * d);
    psiGround.transposeInPlace();
    psiGround.resize(mMax * d * m * d, 1);
};

EffectiveHamiltonian TheBlock::createHSuperFinal(const TheBlock& compBlock,
                                                 int skips) const
{
    int compm = compBlock.m;
    return EffectiveHamiltonian(qNumList, compBlock.qNumList, ham,
                                MatrixXd(kp(kp(hS, Id_d)
                                            + ham.blockAdjacentSiteJoin(1, off0RhoBasisH2)
                                            + ham.blockAdjacentSiteJoin(2, off1RhoBasisH2),
                                            Id(compm * d))
                                         + ham.lBlockrSiteJoin(off0RhoBasisH2, compm)
                                         + ham.siteSiteJoin(m, compm)
                                         + ham.lSiterBlockJoin(m, compBlock.off0RhoBasisH2)
                                         + kp(Id(m * d), ham.blockAdjacentSiteJoin(1, compBlock.off0RhoBasisH2)
                                                         + ham.blockAdjacentSiteJoin(2, compBlock.off1RhoBasisH2)
                                                         + kp(compBlock.hS, Id_d))),
                                m, compm, skips);
};

MatrixXd TheBlock::changeBasis(const MatrixXd& mat) const
{
	return primeToRhoBasis.adjoint() * mat * primeToRhoBasis;
};
