
// #include <IMInclude.h>


#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "linearDecon.h"
// extern "C" {
// #include <clapack.h>  // clapack_sgetrf(), clapack_sgetri()
// }

#ifndef _WIN32
#include <sys/time.h>  //gettimeofday()
#endif

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

void apodize(int napodize, CImg<> &image)
{
  float diff,fact;
  int k,l;

  int nx = image.width();
  int ny = image.height();
  if (nx-ny == 2) // most likely there're extra 2 columns in this case
    nx -= 2;

  for (int z=0; z<image.depth(); z++) {
    for (k=0; k<nx; k++) {
      diff = (image(k, ny-1, z) - image(k, 0, z)) * 0.5;
      for (l=0; l<napodize; l++) {
        fact = 1. - sin(((l+0.5)/napodize)*M_PI*0.5);
        image(k, l, z) += diff*fact;
        image(k, ny-1-l, z) -= diff*fact;
      }
    }
    for (l=0; l<ny; l++) {
      diff = (image(nx-1, l, z) - image(0, l, z) ) * 0.5;
      for(k=0; k<napodize; k++) {
        fact = 1 - sin(((k+0.5)/napodize)*M_PI*0.5);
        image(k, l, z) += diff*fact;
        image(nx-1-k, l, z) -= diff*fact; 
      }
    }
  }
}

int main(int argc, char *argv[])
{
  // int istream_no, ostream_no, otfstream_no;
  int napodize = 10;
  float background;
  float NA=1.2;
  // int nphases = 1;  /* allow input to be 2D SIM datasets */
  ImgParams imgParams;
  float dz_psf, dr_psf;
  float wiener;

  int RL_iters=0;
  bool bCPU = false;
  // bool bPSF = false;
  float deskewAngle=0.0;
  float rotationAngle=0.0;

  // IMAlPrt(0);       /* suppress printout of file header info */

  TIFFSetWarningHandler(NULL);

  std::string ifiles, ofiles, otffiles;
  po::options_description progopts;
  progopts.add_options()
    ("drdata", po::value<float>(&imgParams.dr)->default_value(.104), "image x-y pixel size (um)")
    ("dzdata", po::value<float>(&imgParams.dz)->default_value(.25), "image z step (um)")
    ("drpsf", po::value<float>(&dr_psf)->default_value(.104), "PSF x-y pixel size (um)")
    ("dzpsf", po::value<float>(&dz_psf)->default_value(.1), "PSF z step (um)")
    ("wavelength,l", po::value<float>(&imgParams.wave)->default_value(.525), "emission wavelength (um)")

    ("wiener,w", po::value<float>(&wiener)->default_value(1e-2), "Wiener constant (regularization factor)")
    ("background,b", po::value<float>(&background)->default_value(90.f), "user-supplied background")
    ("NA,n", po::value<float>(&NA)->default_value(1.2), "numerical aperture")
    ("RL", po::value<int>(&RL_iters)->default_value(0), "run Richardson-Lucy how-many iterations")
    ("CPU,C", po::value<bool>(&bCPU)->implicit_value(true), "use CPU code to run R-L")
    ("deskew,D", po::value<float>(&deskewAngle)->default_value(0.0), "Deskew angle; if not 0.0 then perform deskewing before deconv")
    ("rotate,R", po::value<float>(&rotationAngle)->default_value(0.0), "rotation angle; if not 0.0 then perform rotation around y axis after deconv")
    // ("PSF", po::value<bool>(&bPSF)->implicit_value(true), "PSF instead of OTF file is provided")
    ("input-file", po::value<std::string>(&ifiles), "input file")
    ("otf-file", po::value<std::string>(&otffiles), "OTF file")
    ("output-file", po::value<std::string>(&ofiles), "output file")
    ("help,h", "produce help message")
    ;
  po::positional_options_description p;
  p.add("input-file", 1);
  p.add("output-file", 1);
  p.add("otf-file", 1);

/* Parse commandline option */
  po::variables_map varsmap;

  store(po::command_line_parser(argc, argv).
        options(progopts).positional(p).run(), varsmap);
  notify(varsmap);

  if (varsmap.count("help")) {
    std::cout << progopts << "\n";
    return 0;
  }

  if (!ifiles.length()) {
    std::cout << " PSF dataset file name: ";
    std::cin >> ifiles;
  }
  if (!ofiles.length()) {
    std::cout << " Output file name: ";
    std::cin >> ofiles;
  }
  if (!otffiles.length()) {
    std::cout << " OTF file name: ";
    std::cin >> otffiles;
  }

  // bool bUseTIFF = false;
  CImg<> raw_image(ifiles.c_str());

  imgParams.nx = raw_image.width();
  imgParams.ny = raw_image.height();
  imgParams.nz = raw_image.depth();

  CImg<> raw_imageFFT(imgParams.nx+2, imgParams.ny, imgParams.nz);

  CImg<> complexOTF;
  float dkr_otf, dkz_otf;
  // if (!bPSF) {
    // Assuming 3D rotationally averaged OTF
    complexOTF.assign(otffiles.c_str());
    int nr_otf = complexOTF.height();
    int nz_otf = complexOTF.width() / 2;
    dkr_otf = 1/((nr_otf-1)*2 * dr_psf);
    dkz_otf = 1/(nz_otf * dz_psf);
  // }
  // else {
  //   // Do PSF-to-OTF conversion
  //   PSF_to_OTF(otffiles.c_str());
  // }

  fftwf_plan rfftplan=NULL, rfftplan_inv=NULL;
  if (!RL_iters || bCPU) {
    if (!fftwf_init_threads()) { /* one-time initialization required to use threads */
      printf("Error returned by fftwf_init_threads()\n");
    }

    fftwf_plan_with_nthreads(8);

    rfftplan = fftwf_plan_dft_r2c_3d(imgParams.nz, imgParams.ny, imgParams.nx,
                                     raw_image.data(),
                                     (fftwf_complex *) raw_imageFFT.data(),
                                     FFTW_ESTIMATE);

    rfftplan_inv = fftwf_plan_dft_c2r_3d(imgParams.nz, imgParams.ny, imgParams.nx,
                                         (fftwf_complex *) raw_imageFFT.data(),
                                         raw_image.data(),
                                         FFTW_ESTIMATE);
    raw_image -= background;
  }

  float dkx = 1.0/(imgParams.dr * imgParams.nx);
  float dky = 1.0/(imgParams.dr * imgParams.ny);
  float dkz = 1.0/(imgParams.dz * imgParams.nz);
  float rdistcutoff = 2*NA/(imgParams.wave); // lateral resolution limit in 1/um

  // construct deskew matrix (only 2 numbers: slope and offset (if slope<0)
  CPUBuffer deskewMatrix;
  int newXdimension = imgParams.nx;
  if (fabs(deskewAngle) > 0.0) {
    if (deskewAngle <0) deskewAngle += 180.;
    deskewMatrix.resize(2*sizeof(float));
    float *ptr = (float *) deskewMatrix.getPtr();
    ptr[0] = cos(deskewAngle * M_PI/180.) * imgParams.dz / imgParams.dr;
    newXdimension += floor(imgParams.nz * imgParams.dz * 
                           fabs(cos(deskewAngle * M_PI/180.)) / imgParams.dr)/4.; // TODO /4.
    ptr[1] = 0.0;
    if (ptr[0] < 0)
      ptr[1]= newXdimension - imgParams.nx;

    // update z step size:
    imgParams.dz *= sin(deskewAngle * M_PI/180.);

    for (int j=0; j<2; j++)
      printf("%.4f ", ptr[j]);
    printf("\n");
  }

  CPUBuffer rotMatrix;
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

  if (RL_iters) {
    timeval startTime, endTime;
#ifndef _WIN32
    gettimeofday(&startTime, NULL);
#endif

    if (bCPU) {
      raw_image.max(0.f); // background subtraction earlier could have caused negative pixels
      RichardsonLucy(raw_image, imgParams.dr, imgParams.dz,
                     complexOTF, dkr_otf, dkz_otf,
                     rdistcutoff, RL_iters,
                     rfftplan, rfftplan_inv, raw_imageFFT);
    }
    else
      RichardsonLucy_GPU(raw_image, background,
                         imgParams.dr, imgParams.dz,
                         complexOTF, dkr_otf, dkz_otf,
                         rdistcutoff, RL_iters,
                         deskewMatrix, newXdimension,
                         rotMatrix);
#ifndef _WIN32
    gettimeofday(&endTime, NULL);
    printf("%f seconds elapsed\n", (endTime.tv_sec -  startTime.tv_sec) +  (endTime.tv_usec -  startTime.tv_usec)/1.e6);
#endif
  }
  else { // plain 1-step Wiener filtering
    fftwf_execute_dft_r2c(rfftplan, raw_image.data(), (fftwf_complex *) raw_imageFFT.data());

    wienerfilter(raw_imageFFT, 
                 dkx, dky, dkz,
                 complexOTF,
                 dkr_otf, dkz_otf,
                 rdistcutoff, wiener);

    fftwf_execute_dft_c2r(rfftplan_inv, (fftwf_complex *) raw_imageFFT.data(), raw_image.data());
    raw_image /= raw_image.size();
  }
  raw_image.save(ofiles.c_str());

  return 0; 
}
