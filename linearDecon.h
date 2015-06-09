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
  float dr, dz, wave;
};


#include <boost/program_options.hpp>
namespace po = boost::program_options;

//! class fixed_tokens_typed_value
/*!
  For multi-token options, this class allows defining fixed number of arguments
*/
template< typename T, typename charT = char >
class fixed_tokens_typed_value : public po::typed_value< T, charT > {
   unsigned _min, _max;

   typedef po::typed_value< T, charT > base;

 public:

   fixed_tokens_typed_value( T * storeTo, unsigned min, unsigned max ) 
     : _min(min), _max(max), base( storeTo ) {
       base::multitoken();
   }

   virtual base* min_tokens( unsigned min ) {
       _min = min;
       return this;
   }
   unsigned min_tokens() const {return _min;}

   virtual base* max_tokens( unsigned max ) {
       _max = max;
       return this;
   }
   unsigned max_tokens() const {return _max;}

   base* zero_tokens() {
       _min = _max = 0;
       base::zero_tokens();
       return this;
   }
};

template< typename T >
fixed_tokens_typed_value< T > 
fixed_tokens_value(unsigned min, unsigned max) {
    return fixed_tokens_typed_value< T >(0, min, max ); }

template< typename T >
fixed_tokens_typed_value< T > *
fixed_tokens_value(T * t, unsigned min, unsigned max) {
    fixed_tokens_typed_value< T >* r = new
                   fixed_tokens_typed_value< T >(t, min, max);
    return r; }


// std::complex<float> otfinterpolate(std::complex<float> * otf, float kx, float ky, float kz, int nzotf, int nrotf);
// void RichardsonLucy(CImg<> & raw, float dr, float dz, 
//                       CImg<> & otf, float dkr_otf, float dkz_otf, 
//                       float rcutoff, int nIter,
//                       fftwf_plan rfftplan, fftwf_plan rfftplan_inv, CImg<> &fft);

void RichardsonLucy_GPU(CImg<> & raw, float background,
                        GPUBuffer& otf, int nIter,
                        double deskewFactor, int deskewedNx, int extraShift,
                        int napodize,
                        CPUBuffer &rotationMatrix,
                        cufftHandle rfftplanGPU, cufftHandle rfftplanInvGPU,
                        CImg<> & raw_deskewed, cudaDeviceProp* devprop);

CImg<> MaxIntProj(CImg<> &input, int axis);

void transferConstants(int nx, int ny, int nz, int nrotf, int nzotf,
                       float kxscale, float kyscale, float kzscale,
                       float eps, float *otf);
unsigned findOptimalDimension(unsigned inSize, int step=-1);
// void prepareOTFtexture(float * realpart, float * imagpart, int nx, int ny);
void makeOTFarray(GPUBuffer &otfarray, int nx, int ny, int nz);

void backgroundSubtraction_GPU(GPUBuffer &img, int nx, int ny, int nz, float background, unsigned maxGridXdim);

void filterGPU(GPUBuffer &img, int nx, int ny, int nz,
               cufftHandle & rfftplan, cufftHandle & rfftplanInv,
               GPUBuffer &fftBuf,
               GPUBuffer &otf, bool bConj, unsigned maxGridXdim);

void calcLRcore(GPUBuffer &reblurred, GPUBuffer &raw, int nx, int ny, int nz, unsigned maxGridXdim);

void updateCurrEstimate(GPUBuffer &X_k, GPUBuffer &CC, GPUBuffer &Y_k,
                        int nx, int ny, int nz, unsigned maxGridXdim);

void calcCurrPrevDiff(GPUBuffer &X_k, GPUBuffer &Y_k, GPUBuffer &G_kminus1,
                      int nx, int ny, int nz, unsigned maxGridXdim);

double calcAccelFactor(GPUBuffer &G_km1, GPUBuffer &G_km2,
                       int nx, int ny, int nz, float eps);

void updatePrediction(GPUBuffer &Y_k, GPUBuffer &X_k, GPUBuffer &X_kminus1,
                      double lambda, int nx, int ny, int nz, unsigned maxGridXdim);


void deskew_GPU(GPUBuffer &inBuf, int nx, int ny, int nz,
                double deskewFactor, GPUBuffer &outBuf, int newNx, int extraShift);

void rotate_GPU(GPUBuffer &inBuf, int nx, int ny, int nz,
                GPUBuffer &rotMatrix, GPUBuffer &outBuf);

void cropGPU(GPUBuffer &inBuf, int nx, int ny, int nz,
             int new_nx, int new_ny, int new_nz,
             GPUBuffer &outBuf);
double meanAboveBackground_GPU(GPUBuffer &img, int nx, int ny, int nz, unsigned maxGridXdim);
void rescale_GPU(GPUBuffer &img, int nx, int ny, int nz, float scale, unsigned maxGridXdim);
void apodize_GPU(GPUBuffer* image, int nx, int ny, int nz, int napodize);


std::vector<std::string> gatherMatchingFiles(std::string &target_path, std::string &pattern);
std::string makeOutputFilePath(std::string inputFileName, std::string subdir="GPUdecon",
                               std::string insert="_decon");
void makeDeskewedDir(std::string subdirname);



#ifdef _WIN32
#ifdef CUDADECON_IMPORT
  #define CUDADECON_API __declspec( dllimport )
#else
  #define CUDADECON_API __declspec( dllexport )
#endif
#else
  #define CUDADECON_API
#endif

//! All DLL interface calls start HERE:
extern "C" {

//! Call RL_interface_init() as the first step
/*!
 * nx, ny, and nz: raw image dimensions
 * dr: raw image pixel size
 * dz: raw image Z step
 * dr_psf: PSF pixel size
 * dz_psf: PSF Z step
 * deskewAngle: deskewing angle; usually -32.8 on Bi-chang scope and 32.8 on Wes scope
 * rotationAngle: if 0 then no final rotation is done; otherwise set to the same as deskewAngle
 * outputWidth: if set to 0, then calculate the output width because of deskewing; otherwise use this value as the output width
 * OTF_file_name: file name of OTF
*/
CUDADECON_API int RL_interface_init(int nx, int ny, int nz, float dr, float dz, float dr_psf, float dz_psf, float deskewAngle, float rotationAngle, int outputWidth, char * OTF_file_name);

//! RL_interface() to run deconvolution
/*!
 * raw_data: uint16 pointer to raw data buffer
 * nx, ny, and nz: raw image dimensions
 * result: float pointer to pre-allocated result buffer; the results' dimension can be different than raw_data's; see RL_interface_driver.cpp for an example
 * background: camera dark current (~100)
 * nIters: how many iterations to run
 * extraShift: in pixels; sometimes an extra shift in X is needed to center the deskewed image better
*/
CUDADECON_API int RL_interface(const unsigned short * const raw_data, int nx, int ny, int nz, float * const result, float background, int nIters, int extraShift);

//! Call this before program quits to release global GPUBuffer d_interpOTF
CUDADECON_API void RL_cleanup();

//! The following are for retrieving the calculated output dimensions; can be used to allocate result buffer before calling RL_interface()
CUDADECON_API  unsigned get_output_nx();
CUDADECON_API  unsigned get_output_ny();
CUDADECON_API  unsigned get_output_nz();
}

#endif
