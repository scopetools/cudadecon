#include "linearDecon.h"
#include <exception>

std::complex<float> otfinterpolate(std::complex<float> * otf, float kx, float ky, float kz, int nzotf, int nrotf)
// Use sub-pixel coordinates (kx,ky,kz) to linearly interpolate a rotationally-averaged 3D OTF ("otf").
// otf has 2 dimensions: fast dimension is kz with length "nzotf" while the slow dimension is kr.
{
  int irindex, izindex, indices[2][2];
  float krindex, kzindex;
  float ar, az;

  krindex = sqrt(kx*kx + ky*ky);
  kzindex = (kz<0 ? kz+nzotf : kz);

  if (krindex < nrotf-1 && kzindex < nzotf) {
    irindex = floor(krindex);
    izindex = floor(kzindex);

    ar = krindex - irindex;
    az = kzindex - izindex;  // az is always 0 for 2D case, and it'll just become a 1D interp

    if (izindex == nzotf-1) {
      indices[0][0] = irindex*nzotf+izindex;
      indices[0][1] = irindex*nzotf+0;
      indices[1][0] = (irindex+1)*nzotf+izindex;
      indices[1][1] = (irindex+1)*nzotf+0;
    }
    else {
      indices[0][0] = irindex*nzotf+izindex;
      indices[0][1] = irindex*nzotf+(izindex+1);
      indices[1][0] = (irindex+1)*nzotf+izindex;
      indices[1][1] = (irindex+1)*nzotf+(izindex+1);
    }

    return (1-ar)*(otf[indices[0][0]]*(1-az) + otf[indices[0][1]]*az) +
      ar*(otf[indices[1][0]]*(1-az) + otf[indices[1][1]]*az);
  }
  else
    return std::complex<float>(0, 0);
}

int wienerfilter(CImg<> & g, float dkx, float dky, float dkz, 
                 CImg<> & otf, float dkr_otf, float dkz_otf, 
                 float rcutoff, float wiener)
{
  /* 'g' is the raw data's FFT (half kx axis); 
     it is also the result upon return */
  int i, j, k;
  float kz, ky, kx; 
  float amp2, rho, kxscale, kyscale, kzscale, kr;
  std::complex<float> A_star_g, otf_val;
  float w;

  w = wiener*wiener;
  kxscale = dkx/dkr_otf;
  kyscale = dky/dkr_otf;
  kzscale = dkz/dkz_otf;

  int nx = g.width()/2; // '/2' because g is CImg<float> hijacked for complex storage
  int ny = g.height();
  int nz = g.depth();

  std::complex<float> result;

#pragma omp parallel for private(k, i, j, kz, ky, kx, kr, otf_val, amp2, A_star_g, rho, result)
  for (k=0; k<nz; k++) {
    kz = ( k>nz/2 ? k-nz : k );
    for (i=0; i<ny; i++) {
      ky = ( i > ny/2 ? i-ny : i );
      for (j=0; j<nx; j++) {
        kx = j;
        kr = sqrt(kx*kx*dkx*dkx + ky*ky*dky*dky);
        if (kr <=rcutoff) {
          otf_val = otfinterpolate((std::complex<float>*) otf.data(),
                                   kx*kxscale, ky*kyscale,
                                   kz*kzscale, otf.width()/2, otf.height());

          amp2 = otf_val.real() * otf_val.real() + otf_val.imag() * otf_val.imag();
          A_star_g = std::conj(otf_val) * std::complex<float>(g(2*j, i, k), g(2*j+1, i, k));

          /* apodization */
          rho = kr / rcutoff;
          result = A_star_g / (amp2+w) * (1-rho);
          g(2*j, i, k) = result.real();
          g(2*j+1, i, k) = result.imag();
        }
        else {
          g(2*j, i, k) = 0;
          g(2*j+1, i, k) = 0;
        }
      }
    }
  }
  return 0;
}

int main(int argc, char *argv[])
{
  int napodize;
  float background;
  float NA=1.2;
  ImgParams imgParams;
  float dz_psf, dr_psf;
  float wiener;

  int RL_iters=0;
  bool bSaveDeskewedRaw = false;
  float deskewAngle=0.0;
  float rotationAngle=0.0;
  unsigned outputWidth;
  int extraShift=0;
  std::vector<int> final_CropTo_boundaries;
  bool bSaveUshort = false;
  std::vector<bool> bDoMaxIntProj;
  std::vector< CImg<> > MIprojections;

  TIFFSetWarningHandler(NULL);

  std::string datafolder, filenamePattern, otffiles;
  po::options_description progopts;
  progopts.add_options()
    ("drdata", po::value<float>(&imgParams.dr)->default_value(.104), "image x-y pixel size (um)")
    ("dzdata,z", po::value<float>(&imgParams.dz)->default_value(.25), "image z step (um)")
    ("drpsf", po::value<float>(&dr_psf)->default_value(.104), "PSF x-y pixel size (um)")
    ("dzpsf,Z", po::value<float>(&dz_psf)->default_value(.1), "PSF z step (um)")
    ("wavelength,l", po::value<float>(&imgParams.wave)->default_value(.525), "emission wavelength (um)")
    ("wiener,W", po::value<float>(&wiener)->default_value(-1.0), "Wiener constant (regularization factor); if this value is postive then do Wiener filter instead of R-L")
    ("background,b", po::value<float>(&background)->default_value(90.f), "user-supplied background")
    ("napodize,a", po::value<int>(&napodize)->default_value(15), "number of pixels to soften edge with")
    ("NA,n", po::value<float>(&NA)->default_value(1.2), "numerical aperture")
    ("RL,i", po::value<int>(&RL_iters)->default_value(15), "run Richardson-Lucy how-many iterations")
    ("deskew,D", po::value<float>(&deskewAngle)->default_value(0.0), "Deskew angle; if not 0.0 then perform deskewing before deconv")
    ("width,w", po::value<unsigned>(&outputWidth)->default_value(0), "If deskewed, the output image's width")
    ("shift,x", po::value<int>(&extraShift)->default_value(0), "If deskewed, the output image's extra shift in X (positive->left")
    ("rotate,R", po::value<float>(&rotationAngle)->default_value(0.0), "rotation angle; if not 0.0 then perform rotation around y axis after deconv")
    ("saveDeskewedRaw,S", po::bool_switch(&bSaveDeskewedRaw)->default_value(false), "save deskewed raw data to files")
    // ("crop,C", po::value< std::vector<int> >(&final_CropTo_boundaries)->multitoken(), "takes 6 integers separated by space: x1 x2 y1 y2 z1 z2; crop final image size to [x1:x2, y1:y2, z1:z2]")
    // ("MIP,M", po::value< std::vector<bool> >(&bDoMaxIntProj)->multitoken(), "takes 3 binary numbers separated by space to indicate whether save a max-intensity projection along x, y, or z axis")
    ("crop,C", fixed_tokens_value< std::vector<int> >(&final_CropTo_boundaries, 6, 6), "takes 6 integers separated by space: x1 x2 y1 y2 z1 z2; crop final image size to [x1:x2, y1:y2, z1:z2]")
    ("MIP,M", fixed_tokens_value< std::vector<bool> >(&bDoMaxIntProj, 3, 3), "takes 3 binary numbers separated by space to indicate whether save a max-intensity projection along x, y, or z axis")
    ("uint16,u", po::bool_switch(&bSaveUshort)->implicit_value(true), "whether to save result in uint16 format; this should be used only if no actual decon is performed")
    ("input-dir", po::value<std::string>(&datafolder)->required(), "input folder name")
    ("otf-file", po::value<std::string>(&otffiles)->required(), "OTF file")
    ("filename-pattern", po::value<std::string>(&filenamePattern)->required(), "pattern in file names")
    ("help,h", "produce help message")
    ;
  po::positional_options_description p;
  p.add("input-dir", 1);
  p.add("filename-pattern", 1);
  p.add("otf-file", 1);

// Parse commandline option:
  po::variables_map varsmap;
  try {
    store(po::command_line_parser(argc, argv).
          options(progopts).positional(p).run(), varsmap);
    if (varsmap.count("help")) {
      std::cout << progopts << "\n";
      return 0;
    }

    notify(varsmap);

    // if (varsmap.count("crop")) {
    //   if (final_CropTo_boundaries.size() != 6)
    //     throw std::runtime_error("Exactly 6 integers are required for the -C or --crop flag!");
    //   std::copy(final_CropTo_boundaries.begin(), final_CropTo_boundaries.end(), std::ostream_iterator<int>(std::cout, ", "));
    //   std::cout << std::endl;
    // }
    // std::cout << bDoMaxIntProj.size() << std::endl;
  }
  catch (std::exception &e) {
    std::cout << "\n!!Error occurred: " << e.what() << std::endl;
    return 0;
  }

  CImg<> raw_image, raw_imageFFT, complexOTF, raw_deskewed;
  float dkr_otf, dkz_otf;
  float dkx, dky, dkz, rdistcutoff;
  fftwf_plan rfftplan=NULL, rfftplan_inv=NULL;
  CPUBuffer rotMatrix;
  double deskewFactor=0;
  bool bCrop = false;
  unsigned new_ny, new_nz, new_nx;
  int deskewedXdim = 0;
  cufftHandle rfftplanGPU, rfftplanInvGPU;
  GPUBuffer d_interpOTF(0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  // Loop over all matching input TIFFs:
  try {
    // Gather all files in 'datafolder' and matching the file name pattern:
    std::vector< std::string > all_matching_files = 
      gatherMatchingFiles(datafolder, filenamePattern);

    for (std::vector<std::string>::iterator it=all_matching_files.begin();
         it != all_matching_files.end(); it++) {

      std::cout<< *it << std::endl;
      raw_image.assign(it->c_str());

      // If it's the first input file, initialize a bunch including:
      // 1. crop image to make dimensions nice factorizable numbers
      // 2. calculate deskew parameters, new X dimensions
      // 3. calculate rotation matrix
      // 4. create FFT plans
      // 5. transfer constants into GPU device constant memory
      // 6. make 3D OTF array in device memory
      if (it == all_matching_files.begin()) {
        unsigned nx = raw_image.width();
        unsigned ny = raw_image.height();
        unsigned nz = raw_image.depth();

        printf("Original image size: nz=%d, ny=%d, nx=%d\n", nz, ny, nx);

        if (RL_iters>0) {
          new_ny = findOptimalDimension(ny);
          if (new_ny != ny) {
            printf("new ny=%d\n", new_ny);
            bCrop = true;
          }

          new_nz = findOptimalDimension(nz);
          if (new_nz != nz) {
            printf("new nz=%d\n", new_nz);
            bCrop = true;
          }

          // only if no deskewing is happening do we want to change image width here
          new_nx = nx;
          if (!fabs(deskewAngle) > 0.0) {
            new_nx = findOptimalDimension(nx);
            if (new_nx != nx) {
              printf("new nx=%d\n", new_nx);
              bCrop = true;
            }
          }
        }
        else {
          new_nx = nx;
          new_ny = ny;
          new_nz = nz;
        }

        // Load OTF (assuming 3D rotationally averaged OTF):
        complexOTF.assign(otffiles.c_str());
        unsigned nr_otf = complexOTF.height();
        unsigned nz_otf = complexOTF.width() / 2;
        dkr_otf = 1/((nr_otf-1)*2 * dr_psf);
        dkz_otf = 1/(nz_otf * dz_psf);

        // Construct deskew matrix:
        deskewedXdim = new_nx;
        if (fabs(deskewAngle) > 0.0) {
          if (deskewAngle <0) deskewAngle += 180.;
          deskewFactor = cos(deskewAngle * M_PI/180.) * imgParams.dz / imgParams.dr;
          if (outputWidth ==0)
            deskewedXdim += floor(new_nz * imgParams.dz * 
                                  fabs(cos(deskewAngle * M_PI/180.)) / imgParams.dr);
          // TODO: sometimes deskewedXdim calc'ed this way is too large
          else
            deskewedXdim = outputWidth; // use user-provided output width if available

          deskewedXdim = findOptimalDimension(deskewedXdim);

          // update z step size:
          imgParams.dz *= sin(deskewAngle * M_PI/180.);

          printf("deskewFactor=%f, new nx=%d\n", deskewFactor, deskewedXdim);

          if (bSaveDeskewedRaw) {
            raw_deskewed.assign(deskewedXdim, new_ny, new_nz);
            makeDeskewedDir("Deskewed");
          }
        }

        // Construct rotation matrix:
        if (fabs(rotationAngle) > 0.0) {
          rotMatrix.resize(4*sizeof(float));
          rotationAngle *= M_PI/180;
          float stretch = imgParams.dr / imgParams.dz;
          float *p = (float *)rotMatrix.getPtr();
          p[0] = cos(rotationAngle) * stretch;
          p[1] = sin(rotationAngle) * stretch;
          p[2] = -sin(rotationAngle);
          p[3] = cos(rotationAngle);
        }

        if (wiener >0) {
          raw_imageFFT.assign(deskewedXdim+2, new_ny, new_nz);

          if (!fftwf_init_threads()) { /* one-time initialization required to use threads */
            printf("Error returned by fftwf_init_threads()\n");
          }

          fftwf_plan_with_nthreads(8);
          rfftplan = fftwf_plan_dft_r2c_3d(new_nz, new_ny, deskewedXdim,
                                           raw_image.data(),
                                           (fftwf_complex *) raw_imageFFT.data(),
                                           FFTW_ESTIMATE);
          rfftplan_inv = fftwf_plan_dft_c2r_3d(new_nz, new_ny, deskewedXdim,
                                               (fftwf_complex *) raw_imageFFT.data(),
                                               raw_image.data(),
                                               FFTW_ESTIMATE);
        }
        else if (RL_iters>0) {
          // Create reusable cuFFT plans
          cufftResult cuFFTErr = cufftPlan3d(&rfftplanGPU, new_nz, new_ny, deskewedXdim, CUFFT_R2C);
          if (cuFFTErr != CUFFT_SUCCESS) {
            std::cout << "Error code: " << cuFFTErr << std::endl;
            throw std::runtime_error("cufftPlan3d() r2c failed.");
          }
          cuFFTErr = cufftPlan3d(&rfftplanInvGPU, new_nz, new_ny, deskewedXdim, CUFFT_C2R);
          if (cuFFTErr != CUFFT_SUCCESS) {
            std::cout << "Error code: " << cuFFTErr << std::endl;
            throw std::runtime_error("cufftPlan3d() c2r failed.");
          }
        }

        dkx = 1.0/(imgParams.dr * deskewedXdim);
        dky = 1.0/(imgParams.dr * new_ny);
        dkz = 1.0/(imgParams.dz * new_nz);
        rdistcutoff = 2*NA/(imgParams.wave); // lateral resolution limit in 1/um

        // transfer a bunch of constants to device, including OTF array:
        float eps = std::numeric_limits<float>::epsilon();
        transferConstants(deskewedXdim, new_ny, new_nz,
                          complexOTF.height(), complexOTF.width()/2,
                          dkx/dkr_otf, dky/dkr_otf, dkz/dkz_otf,
                          eps, complexOTF.data());

        // make a 3D interpolated OTF array:
        d_interpOTF.resize(new_nz * new_ny * (deskewedXdim+2) * sizeof(float));
        makeOTFarray(d_interpOTF, deskewedXdim, new_ny, new_nz);
      } // if (it == all_matching_files.begin())

      if (bCrop) {
        raw_image.crop(0, 0, 0, 0, new_nx-1, new_ny-1, new_nz-1, 0);
        // If deskew is to happen, it'll be performed inside RichardsonLucy_GPU() on GPU;
        // but here raw data's x dimension is still just "new_nx"
      }

      if (wiener >0) { // plain 1-step Wiener filtering
        raw_image -= background;
        fftwf_execute_dft_r2c(rfftplan, raw_image.data(), (fftwf_complex *) raw_imageFFT.data());

        wienerfilter(raw_imageFFT, 
                     dkx, dky, dkz,
                     complexOTF,
                     dkr_otf, dkz_otf,
                     rdistcutoff, wiener);

        fftwf_execute_dft_c2r(rfftplan_inv, (fftwf_complex *) raw_imageFFT.data(), raw_image.data());
        raw_image /= raw_image.size();
      }
      else if (RL_iters || raw_deskewed.size() || rotMatrix.getSize()) {
        RichardsonLucy_GPU(raw_image, background, d_interpOTF, RL_iters,
                           deskewFactor, deskewedXdim, extraShift, napodize, rotMatrix,
                           rfftplanGPU, rfftplanInvGPU, raw_deskewed, &deviceProp);
      }
      else {
        std::cerr << "Nothing is performed\n";
        break;
      }

      if (! final_CropTo_boundaries.empty()) {
        raw_image.crop(final_CropTo_boundaries[0], final_CropTo_boundaries[2],
                       final_CropTo_boundaries[4], 0,
                       final_CropTo_boundaries[1], final_CropTo_boundaries[3],
                       final_CropTo_boundaries[5], 0);
        if (raw_deskewed.size())
          raw_deskewed.crop(final_CropTo_boundaries[0], final_CropTo_boundaries[2],
                            final_CropTo_boundaries[4], 0,
                            final_CropTo_boundaries[1], final_CropTo_boundaries[3],
                            final_CropTo_boundaries[5], 0);
      }

      if (bSaveDeskewedRaw) {
        if (! bSaveUshort)
          raw_deskewed.save(makeOutputFilePath(*it, "Deskewed", "_deskewed").c_str());
        else {
          CImg<unsigned short> uint16Img(raw_deskewed);
          uint16Img.save(makeOutputFilePath(*it, "Deskewed", "_deskewed").c_str());
          }
      }
      if (RL_iters || rotMatrix.getSize()) {
        if (! bSaveUshort)
          raw_image.save(makeOutputFilePath(*it).c_str());
        else {
          CImg<unsigned short> uint16Img(raw_image);
          uint16Img.save(makeOutputFilePath(*it).c_str());
          }
      }

      if (it == all_matching_files.begin() && bDoMaxIntProj.size())
        makeDeskewedDir("GPUdecon/MIPs");

      if (bDoMaxIntProj.size() == 3) {
        if (bDoMaxIntProj[0]) {
          CImg<> proj = MaxIntProj(raw_image, 0);
          proj.save(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_x").c_str());
        }
        if (bDoMaxIntProj[1]) {
          CImg<> proj = MaxIntProj(raw_image, 1);
          proj.save(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_y").c_str());
        }
        if (bDoMaxIntProj[2]) {
          CImg<> proj = MaxIntProj(raw_image, 2);
          proj.save(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_z").c_str());
        }
      }

    } // iteration over all_matching_files
  } // try {} block
  catch (std::exception &e) {
    std::cerr << "\n!!Error occurred: " << e.what() << std::endl;
    return 0;
  }
  return 0; 
}


CImg<> MaxIntProj(CImg<> &input, int axis)
{
  CImg <> out;

  if (axis==0) {
    CImg<> maxvals(input.height(), input.depth());
    maxvals = -1e10;
#pragma omp parallel for
    cimg_forYZ(input, y, z) for (int x=0; x<input.width(); x++) {
      if (input(x, y, z) > maxvals(y, z))
        maxvals(y, z) = input(x, y, z);
    }
    return maxvals;
  }

  if (axis==1) {
    CImg<> maxvals(input.width(), input.depth());
    maxvals = -1e10;
#pragma omp parallel for
    cimg_forXZ(input, x, z) for (int y=0; y<input.height(); y++) {
      if (input(x, y, z) > maxvals(x, z))
        maxvals(x, z) = input(x, y, z);
    }
    return maxvals;
  }

  if (axis==2) {
    CImg<> maxvals(input.width(), input.height());
    maxvals = -1e10;
#pragma omp parallel for
    cimg_forXY(input, x, y) for (int z=0; z<input.depth(); z++) {
      if (input(x, y, z) > maxvals(x, y))
        maxvals(x, y) = input(x, y, z);
    }
    return maxvals;
  }

  
}
