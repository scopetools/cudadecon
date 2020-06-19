#ifndef LINEAR_DECON_H
#define LINEAR_DECON_H

#include <iomanip> // std::setw
#include <iostream>

#include <complex>
#include <string>
#include <vector>

#ifndef __clang__
#include <omp.h>
#endif

#include <fftw3.h>

#include <cufft.h>

#include <cuda_runtime.h>
#include <helper_cuda.h> //for error cufft error message

#include <CPUBuffer.h>
#include <GPUBuffer.h>
#include <PinnedCPUBuffer.h>

#include <thread> //for asynchronous loading of raw .tif files

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <math.h>

#define cimg_use_tiff
#define cimg_display 0
#include <CImg.h>
using namespace cimg_library;

// CUDA Profiling
#ifndef _WINDLL
//#define USE_CUDNN // uncomment this to use NVIDIA Deep Neural Network
//#define USE_NVTX //uncomment this to use NVIDIA Profiler
#endif

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#ifdef USE_NVTX // https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
#include <cuda_profiler_api.h>
#include <nvToolsExtCudaRt.h>
// How to add to VS :
// https://stackoverflow.com/questions/14717203/use-of-nvidia-tools-extension-under-visual-studio-2010
// Need to have dll from here : C:\Program Files\NVIDIA
// Corporation\NvToolsExt\bin\x64
#include "nvToolsExt.h"

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                                                  \
  {                                                                            \
    int color_id = cid;                                                        \
    color_id = color_id % num_colors;                                          \
    nvtxEventAttributes_t eventAttrib = {0};                                   \
    eventAttrib.version = NVTX_VERSION;                                        \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                          \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                   \
    eventAttrib.color = colors[color_id];                                      \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                         \
    eventAttrib.message.ascii = name;                                          \
    nvtxRangePushEx(&eventAttrib);                                             \
  }
#define POP_RANGE nvtxRangePop();
#define MARKIT(name)                                                           \
  { nvtxMarkA(name); }

#else // If not using NVTX, then just make these do nothing.
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#define MARKIT(name)
#endif

struct ImgParams {
  float dr, dz, wave;
};

#include <boost/program_options.hpp>
namespace po = boost::program_options;

//! class fixed_tokens_typed_value
/*!
  For multi-token options, this class allows defining fixed number of arguments
*/
template <typename T, typename charT = char>
class fixed_tokens_typed_value : public po::typed_value<T, charT> {
  unsigned _min, _max;

  typedef po::typed_value<T, charT> base;

public:
  fixed_tokens_typed_value(T *storeTo, unsigned min, unsigned max)
      : base(storeTo), _min(min), _max(max) {
    base::multitoken();
  }

  virtual base *min_tokens(unsigned min) {
    _min = min;
    return this;
  }
  unsigned min_tokens() const { return _min; }

  virtual base *max_tokens(unsigned max) {
    _max = max;
    return this;
  }
  unsigned max_tokens() const { return _max; }

  base *zero_tokens() {
    _min = _max = 0;
    base::zero_tokens();
    return this;
  }
};

template <typename T>
fixed_tokens_typed_value<T> fixed_tokens_value(unsigned min, unsigned max) {
  return fixed_tokens_typed_value<T>(0, min, max);
}

template <typename T>
fixed_tokens_typed_value<T> *fixed_tokens_value(T *t, unsigned min,
                                                unsigned max) {
  fixed_tokens_typed_value<T> *r = new fixed_tokens_typed_value<T>(t, min, max);
  return r;
}

// std::complex<float> otfinterpolate(std::complex<float> * otf, float kx, float
// ky, float kz, int nzotf, int nrotf); void RichardsonLucy(CImg<> & raw, float
// dr, float dz,
//                       CImg<> & otf, float dkr_otf, float dkz_otf,
//                       float rcutoff, int nIter,
//                       fftwf_plan rfftplan, fftwf_plan rfftplan_inv, CImg<>
//                       &fft);

void RichardsonLucy_GPU(CImg<> &raw, float background, GPUBuffer &otf,
                        int nIter, double deskewFactor, int deskewedNx,
                        int extraShift, int napodize, int nZblend,
                        CPUBuffer &rotationMatrix, cufftHandle rfftplanGPU,
                        cufftHandle rfftplanInvGPU,
                        CImg<> &raw_deskewed, cudaDeviceProp *devprop,
                        bool bFlatStartGuess, float my_median, bool bDoRescale,
                        bool bSkewedDecon,
                        float padVal = 0, bool bDupRevStack = false,
                        bool UseOnlyHostMem = false, int myGPUdevice = 0);

CImg<> MaxIntProj(CImg<> &input, int axis);

void transferConstants(int nx, int ny, int nz, int nxotf, int nyotf, int nzotf,
                       float kxscale, float kyscale, float kzscale, int bLR,
                       float eps);
unsigned findOptimalDimension(unsigned inSize, int step = -1);
// void prepareOTFtexture(float * realpart, float * imagpart, int nx, int ny);
void determine_OTF_dimensions(CImg<> &complexOTF, float dr_psf, float dz_psf,
                              unsigned &nx_otf, unsigned &ny_otf, unsigned &nz_otf,
                              float &dkx_otf, float &dky_otf, float &dkz_otf);
void makeOTFarray(GPUBuffer &raw_otfarray, GPUBuffer &otfarray, int nx, int ny, int nz);

void backgroundSubtraction_GPU(GPUBuffer &img, int nx, int ny, int nz,
                               float background, unsigned maxGridXdim);

void filterGPU(GPUBuffer &img, int nx, int ny, int nz, cufftHandle &rfftplan,
               cufftHandle &rfftplanInv, cufftHandle &, GPUBuffer &fftBuf, GPUBuffer &otf,
               bool bConj, unsigned maxGridXdim);

void calcLRcore(GPUBuffer &reblurred, GPUBuffer &raw, int nx, int ny, int nz,
                unsigned maxGridXdim);

void updateCurrEstimate(GPUBuffer &X_k, GPUBuffer &CC, GPUBuffer &Y_k, int nx,
                        int ny, int nz, unsigned maxGridXdim);

void calcCurrPrevDiff(GPUBuffer &X_k, GPUBuffer &Y_k, GPUBuffer &G_kminus1,
                      int nx, int ny, int nz, unsigned maxGridXdim);

double calcAccelFactor(GPUBuffer &G_km1, GPUBuffer &G_km2, int nx, int ny,
                       int nz, float eps, int myGPUdevice);

void updatePrediction(GPUBuffer &Y_k, GPUBuffer &X_k, GPUBuffer &X_kminus1,
                      double lambda, int nx, int ny, int nz,
                      unsigned maxGridXdim);

void deskew_GPU(GPUBuffer &inBuf, int nx, int ny, int nz, double deskewFactor,
                GPUBuffer &outBuf, int newNx, int extraShift, float padVal = 0);

void rotate_GPU(GPUBuffer &inBuf, int nx, int ny, int nz, GPUBuffer &rotMatrix,
                GPUBuffer &outBuf, int nx_out, int nz_out);

void affine_GPU(cudaArray *cuArray, int nx, int ny, int nz, float *result,
                GPUBuffer &affMat);

void affine_GPU_RA(cudaArray *cuArray, int nx, int ny, int nz, float dx,
                   float dy, float dz, float *result, GPUBuffer &affMat);

void camcor_GPU(int nx, int ny, int nz, GPUBuffer &outBuf);

void setupConst(int nx, int ny, int nz);

void setupCamCor(int nx, int ny, float *h_caparam);

void setupData(int nx, int ny, int nz, unsigned *h_data);

void cropGPU(GPUBuffer &inBuf, int nx, int ny, int nz, int new_nx, int new_ny,
             int new_nz, GPUBuffer &outBuf);
double meanAboveBackground_GPU(GPUBuffer &img, int nx, int ny, int nz,
                               unsigned maxGridXdim, int myGPUdevice);
void rescale_GPU(GPUBuffer &img, int nx, int ny, int nz, float scale,
                 unsigned maxGridXdim);
void apodize_GPU(GPUBuffer &image, int nx, int ny, int nz, int napodize);
void zBlend_GPU(GPUBuffer &image, int nx, int ny, int nz, int nZblend);

void duplicateReversedStack_GPU(GPUBuffer &in, int nx, int ny, int nz);

std::vector<std::string> gatherMatchingFiles(std::string &target_path,
                                             std::string &pattern,
                                             bool no_overwrite);
std::string makeOutputFilePath(std::string inputFileName,
                               std::string subdir = "GPUdecon",
                               std::string insert = "_decon");
void makeNewDir(std::string subdirname);

#ifdef _WIN32
#ifdef CUDADECON_IMPORT
#define CUDADECON_API __declspec(dllimport)
#else
#define CUDADECON_API __declspec(dllexport)
#endif
#else
#define CUDADECON_API
#endif

    // ***************************************************************
    //                   SHARED LIBRARY CALLS                        *
    // ***************************************************************

    extern "C" {

  //! Call RL_interface_init() as the first step
  /*!
   * nx, ny, and nz: raw image dimensions
   * dr: raw image pixel size
   * dz: raw image Z step
   * dr_psf: PSF pixel size
   * dz_psf: PSF Z step
   * deskewAngle: deskewing angle; usually -32.8 on Bi-chang scope and 32.8 on
   Wes scope
   * rotationAngle: if 0 then no final rotation is done;
     otherwise set to the same as deskewAngle
   * outputWidth: if set to 0, then calculate the output width because of
   deskewing; otherwise use this value as the output width
   * OTF_file_name: file name of OTF
  */
  CUDADECON_API int RL_interface_init(int nx, int ny, int nz, float dr,
                                      float dz, float dr_psf, float dz_psf,
                                      float deskewAngle, float rotationAngle,
                                      int outputWidth, bool bSkewedDecon,
                                      bool bNoLimitRatio,
                                      char *OTF_file_name);

  //! RL_interface() to run deconvolution
  /*!
   * raw_data: uint16 pointer to raw data buffer
   * nx, ny, and nz: raw image dimensions
   * result: float pointer to pre-allocated result buffer;
     the results' dimension can be different than raw_data's;
     see RL_interface_driver.cpp for an example
   * background: camera dark current (~100)
   * nIters: how many iterations to run
   * extraShift: in pixels; sometimes an extra shift in X is
     needed to center the deskewed image better
  */
  CUDADECON_API int RL_interface(
      const unsigned short *const raw_data, int nx, int ny, int nz,
      float *result, float *raw_deskewed_result, float background,
      bool bDoRescale, bool bSaveDeskewedRaw, int nIters, int extraShift,
      bool bSkewedDecon, int napodize = 0, int nZblend = 0, float padVal = 0,
      bool bDupRevStack = false);

  CUDADECON_API int Deskew_interface(
      const float *const raw_data, int nx, int ny, int nz, float dz, float dr,
      float deskewAngle, float *const result, int outputWidth, int extraShift,
      float padVal = 0);

  CUDADECON_API int Affine_interface(const float *const raw_data, int nx,
                                     int ny, int nz, float *const result,
                                     const float *affMat);

  CUDADECON_API int Affine_interface_RA(
      const float *const raw_data, int nx, int ny, int nz, float dx, float dy,
      float dz, float *const result, const float *affMat);

  CUDADECON_API int camcor_interface_init(int nx, int ny, int nz,
                                          const float *const camparam);

  CUDADECON_API int camcor_interface(const unsigned short *const raw_data,
                                     int nx, int ny, int nz,
                                     unsigned short *const result);

  //! Call this before program quits to release global GPUBuffer d_interpOTF
  CUDADECON_API void RL_cleanup();
  CUDADECON_API void cuda_reset();

  //! The following are for retrieving the calculated output dimensions;
  //  can be used to allocate result buffer before calling RL_interface()
  CUDADECON_API unsigned get_output_nx();
  CUDADECON_API unsigned get_output_ny();
  CUDADECON_API unsigned get_output_nz();
}

#endif
