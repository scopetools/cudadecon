
//#include <vector>
#include <iostream>
#include <complex>

//#include <stdio.h>

#include <fftw3.h>

//#include <tiffio.h>

#define cimg_use_tiff
#define cimg_display 0
#include <CImg.h>
using namespace cimg_library;

#ifdef _WIN32
#define _USE_MATH_DEFINES
#define rint(fp) (int)((fp) >= 0 ? (fp) + 0.5 : (fp) - 0.5)
#endif

#include <math.h>



extern int save_tiff(TIFF *tif, const unsigned int directory, int colind, const int nwaves, int width, int height, float * buffer);
extern int load_tiff(TIFF *const tif, const unsigned int directory, const unsigned colind, float *const buffer);

float fitparabola(float a1, float a2, float a3);
void determine_center_and_background(float *stack5phases, int nx, int ny, int nz, float *xc, float *yc, float *zc, float *background);
void shift_center(std::complex<float> *bands, int nx, int ny, int nz, float xc, float yc, float zc);
void cleanup(std::complex<float> *otfkxkz, int nx, int nz, float dkr, float dkz, float linespacing, int lambdanm, int twolens, float NA, float NIMM);
void radialft(std::complex<float> *bands, int nx, int ny, int nz, std::complex<float> *avg);

bool fixorigin(std::complex<float> *otfkxkz, int nx, int nz, int kx2);
void rescale(std::complex<float> *otfkxkz, int nx, int nz);


#ifdef _WIN32
#ifdef RADIALFT_IMPORT
  #define RADIALFT_API __declspec( dllimport )
#else
  #define RADIALFT_API __declspec( dllexport )
#endif
#else
  #define RADIALFT_API
#endif

extern "C" {
RADIALFT_API int makeOTF(const char *const ifiles, const char *const ofiles,
      int lambdanm = 520, float dz = 0.102, int interpkr = 10,
      bool bUserBackground = false, float background = 90,
      float NA = 1.25, float NIMM = 1.3, float dr = 0.102,
      int krmax = 0, bool bDoCleanup = false);
}



//   ("na", po::value<float>(&NA)->default_value(1.25), "NA of detection objective")
//   ("nimm", po::value<float>(&NIMM)->default_value(1.3), "refractive index of immersion medium")
//   ("xyres", po::value<float>(&dr)->default_value(.102), "x-y pixel size")
//   ("zres", po::value<float>(&dz)->default_value(.102), "z pixel size")
//   ("wavelength", po::value<int>(&lambdanm)->default_value(520), "emission wavelength in nm")
//   ("fixorigin", po::value<int>(&interpkr)->default_value(10),
//    "for all kz, extrapolate using pixels kr=1 to this pixel to get value for kr=0")
//   ("krmax", po::value<int>(&krmax)->default_value(0),
//    "pixels outside this limit will be zeroed (overwriting estimated value from NA and NIMM)")
//   ("nocleanup", po::bool_switch(&bDoCleanup)->implicit_value(false), "elect not to do clean-up outside OTF support")
//   ("background", po::value<float>(&background), "use user-supplied background instead of the estimated")
//   ("input-file", po::value<std::string>(&ifiles)->required(), "input file")
//   ("output-file", po::value<std::string>(&ofiles)->required(), "output file")


std::string ifiles, ofiles;
int nx, ny, nz, nxy;
int i, j, z;
float dkr, dkz, background, estBackground, xcofm, ycofm, zcofm;
float *floatimage;
std::complex<float> *bands, *avg_output;
fftwf_plan rfftplan3d;

// float dr = 0.102;
// float dz = 0.102;
// int interpkr = 10;
// float NA = 1.25;
// float NIMM = 1.3;
// int lambdanm = 520;
// int krmax = 0;
// bool bDoCleanup = true;
// bool bUserBackground = false;

int makeOTF(const char *const ifiles, const char *const ofiles, int lambdanm,
    float dz, int interpkr, bool bUserBackground, float background,
    float NA, float NIMM, float dr, int krmax, bool bDoCleanup)
{
  printf("called");

  TIFFSetWarningHandler(NULL);

  CImg<> rawtiff(ifiles);

  nz = rawtiff.depth();
  ny = rawtiff.height();
  nx = rawtiff.width();

  printf("nx=%d, ny=%d, nz=%d\n", nx, ny, nz);

  dkr = 1/(ny*dr);
  dkz = 1/(nz*dz);

  nxy=(nx+2)*ny;

  floatimage = (float *) malloc(nxy*nz*sizeof(float));
  bands = (std::complex<float> *) floatimage;

 // printf("Reading data...\n\n");

  for(z=0; z<nz; z++)
    for (i=0; i<ny; i++) {
      for (j=0; j<nx; j++)
        floatimage[z*nxy+i*(nx+2)+j] = rawtiff(j, i, z);
    }

  /* Before FFT, estimate bead center position */
  determine_center_and_background(floatimage, nx, ny, nz, &xcofm, &ycofm, &zcofm, &estBackground);

  printf("Center of mass is (%.3f, %.3f, %.3f)\n", xcofm, ycofm, zcofm);

  if (!bUserBackground)
    background = estBackground;

  printf("Background is %.3f\n", background);

  for(z=0; z<nz; z++) {
    for(i=0;i<ny;i++)
      for(j=0;j<nx;j++)
        floatimage[z*nxy + i*(nx+2) + j] -= background;
  }

  rfftplan3d = fftwf_plan_dft_r2c_3d(nz, ny, nx, floatimage,
                                     (fftwf_complex *) floatimage, FFTW_ESTIMATE);

  //printf("Before fft\n");

  fftwf_execute_dft_r2c(rfftplan3d, floatimage, (fftwf_complex *) floatimage);

  fftwf_destroy_plan(rfftplan3d);

  //printf("After fft\n\n");

  /* modify the phase of bands, so that it corresponds to FFT of a bead at origin */
  //printf("Shifting center...\n");

  shift_center(bands, nx, ny, nz, xcofm, ycofm, zcofm);

  CImg<> output_tiff(nz*2, nx/2+1, 1, 1, 0.f);
  avg_output = (std::complex<float> *) output_tiff.data();

  radialft(bands, nx, ny, nz, avg_output);

  if (bDoCleanup)
    cleanup(avg_output, nx, nz, dkr, dkz, 1.0, lambdanm, krmax, NA, NIMM);

  if (interpkr > 0)
    try {
    while (!fixorigin(avg_output, nx, nz, interpkr)) {
      interpkr --;
      if (interpkr < 4)
        throw std::runtime_error("#pixels < 4 used in kr=0 extrapolation");
    }}
  catch (std::exception &e) {
    std::cout << "\n!!Error occurred: " << e.what() << std::endl;
    return 1;
  }

//  printf("%d\n", interpkr);
  rescale(avg_output, nx, nz);

  /* For side bands, combine bandre's and bandim's into bandplus */
  /* Shouldn't this be done later on the averaged bands? */

  output_tiff.save_tiff(ofiles);

  return 0;
}

/*  locate peak pixel to subpixel accuracy by fitting parabolas  */
void determine_center_and_background(float *stack3d, int nx, int ny, int nz, float *xc, float *yc, float *zc, float *background)
{
  int i, j, k, maxi, maxj, maxk, ind, nxy2, infocus_sec;
  int iminus, iplus, jminus, jplus, kminus, kplus;
  float maxval, reval, valminus, valplus;
  double sum;

  //printf("In determine_center_and_background()\n");
  nxy2 = (nx+2)*ny;

  /* Search for the peak pixel */
  /* Be aware that stack3d is of dimension (nx+2)xnyxnz */
  maxval=0.0;
  for(k=0;k<nz;k++)
    for(i=0;i<ny;i++)
      for(j=0;j<nx;j++) {
    ind=k*nxy2+i*(nx+2)+j;
    reval=stack3d[ind];
    if( reval > maxval ) {
      maxval = reval;
      maxi=i; maxj=j;
      maxk=k;
    }
      }

  iminus = maxi-1; iplus = maxi+1;
  if( iminus<0 ) iminus+=ny;
  if( iplus>=ny ) iplus-=ny;
  jminus = maxj-1; jplus = maxj+1;
  if( jminus<0 ) jminus+=nx;
  if( jplus>=nx ) jplus-=nx;
  kminus = maxk-1; kplus = maxk+1;
  if( kminus<0 ) kminus+=nz;
  if( kplus>=nz ) kplus-=nz;

  valminus = stack3d[kminus*nxy2+maxi*(nx+2)+maxj];
  valplus  = stack3d[kplus *nxy2+maxi*(nx+2)+maxj];
  *zc = maxk + fitparabola(valminus, maxval, valplus);

  *zc += 0.6;

  valminus = stack3d[maxk*nxy2+iminus*(nx+2)+maxj];
  valplus  = stack3d[maxk*nxy2+iplus *(nx+2)+maxj];
  *yc = maxi + fitparabola(valminus, maxval, valplus);

  valminus = stack3d[maxk*nxy2+maxi*(nx+2)+jminus];
  valplus  = stack3d[maxk*nxy2+maxi*(nx+2)+jplus];
  *xc = maxj + fitparabola(valminus, maxval, valplus);

  sum = 0;
  infocus_sec = floor(*zc);
  for (i=0; i<*yc-20; i++)
    for (j=0; j<nx; j++)
    sum += stack3d[infocus_sec*nxy2 + i*(nx+2) + j];
  *background = sum / ((*yc-20)*nx);
}

/***************************** fitparabola **********************************/
/*     Fits a parabola to the three points (-1,a1), (0,a2), and (1,a3).     */
/*     Returns the x-value of the max (or min) of the parabola.             */
/****************************************************************************/

float fitparabola( float a1, float a2, float a3 )
{
 float slope,curve,peak;

 slope = 0.5* (a3-a1);         /* the slope at (x=0). */
 curve = (a3+a1) - 2*a2;       /* (a3-a2)-(a2-a1). The change in slope per unit of x. */
 if( curve == 0 )
 {
   printf("no peak: a1=%f, a2=%f, a3=%f, slope=%f, curvature=%f\n",a1,a2,a3,slope,curve);
   return( 0.0 );
 }
 peak = -slope/curve;          /* the x value where slope = 0  */
 if( peak>1.5 || peak<-1.5 )
 {
   printf("bad peak position: a1=%f, a2=%f, a3=%f, slope=%f, curvature=%f, peak=%f\n",a1,a2,a3,slope,curve,peak);
   return( 0.0 );
 }
 return( peak );
}


/* To get rid of checkerboard effect in the OTF bands */
/* (xc, yc, zc) is the estimated center of the point source, which in most cases is the bead */
/* Converted from Fortran code. kz is treated differently than kx and ky. don't know why */
void shift_center(std::complex<float> *bands, int nx, int ny, int nz, float xc, float yc, float zc)
{
  int kin, iin, jin, indin, nxy, kz, kx, ky, kycent, kxcent, kzcent;
  std::complex<float> exp_iphi;
  float phi1, phi2, phi, dphiz, dphiy, dphix;

  kycent = ny/2;
  kxcent = nx/2;
  kzcent = nz/2;
  nxy = (nx/2+1)*ny;

  dphiz = 2*M_PI*zc/nz;
  dphiy = 2*M_PI*yc/ny;
  dphix = 2*M_PI*xc/nx;

  for (kin=0; kin<nz; kin++) {    /* the origin of Fourier space is at (0,0) */
    kz = kin;
    if (kz>kzcent) kz -= nz;
    phi1 = dphiz*kz;      /* first part of phi */
    for (iin=0; iin<ny; iin++) {
      ky = iin;
      if (iin>kycent) ky -= ny;
      phi2 = dphiy*ky;   /* second part of phi */
      for (jin=0; jin<kxcent+1; jin++) {
        kx = jin;
        indin = kin*nxy+iin*(nx/2+1)+jin;
        phi = phi1+phi2+dphix*kx;  /* third part of phi */
        /* kz part of Phi has a minus sign, I don't know why. */
        exp_iphi = std::complex<float> (cos(phi), sin(phi));
        bands[indin] = bands[indin] * exp_iphi;
      }
    }
  }
}

void radialft(std::complex<float> *band, int nx, int ny, int nz, std::complex<float> *avg_output)
{
  int kin, iin, jin, indin, indout, indout_conj, kz, kx, ky, kycent, kxcent, kzcent;
  int *count, nxz, nxy;
  float rdist;

  printf("In radialft()\n");
  kycent = ny/2;
  kxcent = nx/2;
  kzcent = nz/2;
  nxy = (nx/2+1)*ny;
  nxz = (nx/2+1)*nz;

  count = (int *) calloc(nxz, sizeof(int));

  if (!count) {
    printf("No memory availale in radialft()\n");
    exit(-1);
  }

  for (kin=0; kin<nz; kin++) {
    kz = kin;
    if (kin>kzcent) kz -= nz;
    for (iin=0; iin<ny; iin++) {
      ky = iin;
      if (iin>kycent) ky -= ny;
      for (jin=0; jin<kxcent+1; jin++) {
        kx = jin;
        rdist = sqrt(kx*kx+ky*ky);
        if (rdist < nx/2+1) {
          indin = kin*nxy+iin*(nx/2+1)+jin;
          indout = rint(rdist)*nz+kin;
          if (indout < nxz) {
            avg_output[indout] += band[indin];
            count[indout] ++;
          }
          // printf("kz=%d, ky=%d, kx=%d, indout=%d\n", kz, ky, kx, indout);
        }
      }
    }
  }

  for (indout=0; indout<nxz; indout++) {
    if (count[indout]>0) {
      avg_output[indout] /= count[indout];
    }
  }

  /* Then complete the rotational averaging and scaling*/
  for (kx=0; kx<nx/2+1; kx++) {
    indout = kx*nz+0;
    avg_output[indout] = std::complex<float>(avg_output[indout].real(), 0);
    for (kz=1; kz<=nz/2; kz++) {
      indout = kx*nz+kz;
      indout_conj = kx*nz + (nz-kz);
      avg_output[indout] = (avg_output[indout] + conj(avg_output[indout_conj])) / 2.f;
      avg_output[indout_conj] = conj(avg_output[indout]);
    }
  }
  free(count);
}

void cleanup(std::complex<float> *otfkxkz, int nx, int nz, float dkr, float dkz, float linespacing, int lamdanm, int krmax_user, float NA, float NIMM)
{
  int ix, iz, kzstart, kzend, icleanup=nx/2+1;
  float lamda, sinalpha, cosalpha, kr, krmax, beta, kzedge;


  lamda = lamdanm * 0.001;
  sinalpha = NA/NIMM;
  cosalpha = cos(asin(sinalpha));
  krmax = 2*NA/lamda;
  if (krmax_user*dkr<krmax && krmax_user!=0)
    krmax = krmax_user*dkr;

  printf("krmax=%f, lambda=%f\n", krmax, lamda);
  for (ix=0; ix<icleanup; ix++) {
    kr = ix * dkr;
    if ( kr <= krmax ) {
      beta = asin( ( NA - kr*lamda ) /NIMM );
      kzedge = (NIMM/lamda) * ( cos(beta) - cosalpha );
      /* kzstart = floor((kzedge/dkz) + 1.999); */ /* In fortran, it's 2.999 */
      kzstart = rint((kzedge/dkz) + 1);
      kzend = nz - kzstart;
      for (iz=kzstart; iz<=kzend; iz++)
        otfkxkz[ix*nz+iz] = 0;
    }
    else {   /* outside of lateral resolution limit */
      for (iz=0; iz<nz; iz++)
        otfkxkz[ix*nz+iz] = 0;
    }
  }
}

bool fixorigin(std::complex<float> *otfkxkz, int nx, int nz, int kx2)
{
  // linear fit the value at kx=0 using kx in [1, kx2]
  double mean_kx = (kx2+1)/2.; // the mean of [1, 2, ..., n] is (n+1)/2

  // printf("In fixorigin(), kx2=%d\n", nx, nz, kx2);

  for (int z=0; z<nz; z++) {
    std::complex<double> mean_val=0;
    std::complex<double> slope_numerator=0;
    std::complex<double> slope_denominat=0;

    for (int x=1; x<=kx2; x++)
      mean_val += otfkxkz[x*nz + z];

    mean_val /= kx2;
    for (int x=1; x<=kx2; x++) {
      std::complex<double> complexval = otfkxkz [x*nz+z];
      slope_numerator += (x - mean_kx) * (complexval - mean_val);
      slope_denominat += (x - mean_kx) * (x - mean_kx);
    }
    std::complex<double> slope = slope_numerator / slope_denominat;
    otfkxkz[z] = mean_val - slope * mean_kx;  // intercept at kx=0
    if (z==0 && std::abs(otfkxkz[z]) <= std::abs(otfkxkz[nz+z])) {
      return false; // indicating kx2 may be too large
    }
  }
  return true;
}

void rescale(std::complex<float> *otfkxkz, int nx, int nz)
{
  int nxz, ind;
  float valmax=0, mag, scalefactor;

  nxz = (nx/2+1)*nz;
  for (ind=0; ind<nxz; ind++) {
    mag = abs(otfkxkz[ind]);
    if (mag > valmax)
      valmax = mag;
  }
  scalefactor = 1/valmax;

  for (ind=0; ind<nxz; ind++) {
    otfkxkz[ind] *= scalefactor;
  }
}



