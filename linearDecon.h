#ifndef LINEAR_DECON_H
#define LINEAR_DECON_H

#include <iostream>

#include <string>
#include <complex>
#include <vector>

#include <omp.h>

#include <fftw3.h>

#include <cufft.h>

#include <CPUBuffer.h>
#include <GPUBuffer.h>
#include <PinnedCPUBuffer.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <math.h>

#define cimg_use_tiff
#include <CImg.h>
using namespace cimg_library;

struct ImgParams {
//  int nx, ny, nz, nt;
  float dr, dz, wave;
};


std::complex<float> otfinterpolate(std::complex<float> * otf, float kx, float ky, float kz, int nzotf, int nrotf);
void RichardsonLucy(CImg<> & raw, float dr, float dz, 
                      CImg<> & otf, float dkr_otf, float dkz_otf, 
                      float rcutoff, int nIter,
                      fftwf_plan rfftplan, fftwf_plan rfftplan_inv, CImg<> &fft);

void RichardsonLucy_GPU(CImg<> & raw, float background,
                        GPUBuffer& otf, int nIter,
                        CPUBuffer &deskewMatrix, int deskewedNx,
                        CPUBuffer &rotationMatrix,
                        cufftHandle rfftplanGPU, cufftHandle rfftplanInvGPU);

void transferConstants(int nx, int ny, int nz, int nrotf, int nzotf,
                       float kxscale, float kyscale, float kzscale,
                       float eps, float *otf);
unsigned findOptimalDimension(unsigned inSize, int step=-1);
// void prepareOTFtexture(float * realpart, float * imagpart, int nx, int ny);
void makeOTFarray(GPUBuffer &otfarray, int nx, int ny, int nz);

void backgroundSubtraction_GPU(GPUBuffer &img, int nx, int ny, int nz, float background);

void filterGPU(GPUBuffer &img, int nx, int ny, int nz,
               cufftHandle & rfftplan, cufftHandle & rfftplanInv,
               GPUBuffer &fftBuf,
               GPUBuffer &otf, bool bConj);

void calcLRcore(GPUBuffer &reblurred, GPUBuffer &raw, int nx, int ny, int nz);

void updateCurrEstimate(GPUBuffer &X_k, GPUBuffer &CC, GPUBuffer &Y_k,
                        int nx, int ny, int nz);

void calcCurrPrevDiff(GPUBuffer &X_k, GPUBuffer &Y_k, GPUBuffer &G_kminus1,
                      int nx, int ny, int nz);

double calcAccelFactor(GPUBuffer &G_km1, GPUBuffer &G_km2,
                       int nx, int ny, int nz, float eps);

void updatePrediction(GPUBuffer &Y_k, GPUBuffer &X_k, GPUBuffer &X_kminus1,
                      double lambda, int nx, int ny, int nz);


void deskew_GPU(GPUBuffer &inBuf, int nx, int ny, int nz,
                GPUBuffer &deskewMatrix, GPUBuffer &outBuf, int newNx);

void rotate_GPU(GPUBuffer &inBuf, int nx, int ny, int nz,
                GPUBuffer &rotMatrix, GPUBuffer &outBuf);

void cropGPU(GPUBuffer &inBuf, int nx, int ny, int nz,
             int new_nx, int new_ny, int new_nz,
             GPUBuffer &outBuf);

std::vector<std::string> gatherMatchingFiles(std::string &target_path, std::string &pattern);
std::string makeOutputFilePath(std::string inputFileName, std::string insert=std::string("_decon"));

#endif
