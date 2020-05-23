#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <vector>
#include <iostream>
#include <complex>

#include <fftw3.h>

#include <tiffio.h>

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
void fixorigin(CImg<> &otf3d, int kx2);
void rescale(std::complex<float> *otfkxkz, int nx, int nz);
void rescale(CImg<> &otf3d);


int main(int argc, char **argv)
{
#ifdef _WIN32
  HANDLE  hConsole;
  hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

  SetConsoleTextAttribute(hConsole, 11); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
  printf("Created at Howard Hughes Medical Institute Janelia Research Campus. Copyright 2017. All rights reserved.\n");
#ifdef _WIN32
  SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
  std::string ifiles, ofiles;
  int nx, ny, nz, nxy;
  int i, j, z;
  float dr, dz, dkr, dkz, background, estBackground, xcofm, ycofm, zcofm;
  float *floatimage;
  std::complex<float> *bands, *avg_output;
  fftwf_plan rfftplan3d;

  // std::vector<int> interpkr(2);
  int interpkr;
  float NA, NIMM;
  int lambdanm;
  int krmax=0;
  bool bDoCleanup = true;
  bool bUserBackground = false;
  bool b3Dout = false;

  po::options_description progopts;
  progopts.add_options()
    ("na", po::value<float>(&NA)->default_value(1.25), "NA of detection objective")
    ("nimm", po::value<float>(&NIMM)->default_value(1.3), "refractive index of immersion medium")
    ("xyres", po::value<float>(&dr)->default_value(.104), "x-y pixel size")
    ("zres", po::value<float>(&dz)->default_value(.1), "z pixel size")
    ("wavelength", po::value<int>(&lambdanm)->default_value(530), "emission wavelength in nm")
    ("fixorigin", po::value<int>(&interpkr)->default_value(5),
     "for all kz, extrapolate using pixels kr=1 to this pixel to get value for kr=0")
    ("krmax", po::value<int>(&krmax)->default_value(0),
     "pixels outside this limit will be zeroed (overwriting estimated value from NA and NIMM)")
    ("nocleanup", po::bool_switch(&bDoCleanup)->implicit_value(false), "elect not to do clean-up outside OTF support")
    ("background", po::value<float>(&background), "use user-supplied background instead of the estimated")
    ("3Dout,3", po::bool_switch(&b3Dout)->implicit_value(true),
     "Output 3D, instead of rotationally averaged, OTF")
    ("input-file", po::value<std::string>(&ifiles)->required(), "input PSF file")
    ("output-file", po::value<std::string>(&ofiles)->required(), "output OTF file to write")
    ("help,h", "produce help message")
    ;

  po::positional_options_description p;
  p.add("input-file", 1);
  p.add("output-file", 1);

  /* Parse commandline option */
  po::variables_map varsmap;

  store(po::command_line_parser(argc, argv).
        options(progopts).positional(p).run(), varsmap);

  if (argc == 1)  { //if no arguments, show help.
    std::cout << progopts << "\n";
    return 0;
  }

  if (varsmap.count("help")) {
    std::cout << progopts << "\n";
    return 0;
  }

  notify(varsmap);

  if (varsmap.count("background")) {
    bUserBackground = true;
  }

  if (!ifiles.length()) {
    printf(" PSF dataset file name: ");
    std::cin >> ifiles;
  }
  if (!ofiles.length()) {
    printf(" Output OTF file name: ");
    std::cin >> ofiles;
  }

  printf ("bDoCleanup=%i\n", bDoCleanup);
  TIFFSetWarningHandler(NULL);

  CImg<> rawtiff(ifiles.c_str());

  nz = rawtiff.depth();
  ny = rawtiff.height();
  nx = rawtiff.width();

  printf("nx=%d, ny=%d, nz=%d\n", nx, ny, nz);

  dkr = 1/(ny*dr);
  dkz = 1/(nz*dz);

  int nx2 = (nx/2+1)*2;   // in case nx is odd numbered, nx2 is nx+1, not nx+2
  nxy=nx2*ny;

  floatimage = (float *) malloc(nxy*nz*sizeof(float));
  bands = (std::complex<float> *) floatimage;

  printf("Reading data...\n\n");

  for(z=0; z<nz; z++)
    for (i=0; i<ny; i++) {
      for (j=0; j<nx; j++)
        floatimage[z*nxy+i*nx2+j] = rawtiff(j, i, z);
    }

  /* Before FFT, estimate bead center position */
  determine_center_and_background(floatimage, nx, ny, nz, &xcofm, &ycofm, &zcofm, &estBackground);

  printf("Center of mass is (%.3f, %.3f, %.3f)\n\n", xcofm, ycofm, zcofm);

  if (!bUserBackground)
    background = estBackground;
  printf("Background is %.3f\n", background);

  for(z=0; z<nz; z++) {
    for(i=0;i<ny;i++)
      for(j=0;j<nx;j++)
        floatimage[z*nxy + i*nx2 + j] -= background;
  }

  rfftplan3d = fftwf_plan_dft_r2c_3d(nz, ny, nx, floatimage,
                                     (fftwf_complex *) floatimage, FFTW_ESTIMATE);
  printf("Before fft\n");
  fftwf_execute_dft_r2c(rfftplan3d, floatimage, (fftwf_complex *) floatimage);

  fftwf_destroy_plan(rfftplan3d);

  printf("After fft\n\n");

  /* modify the phase of bands, so that it corresponds to FFT of a bead at origin */
  printf("Shifting center...\n");
  shift_center(bands, nx, ny, nz, xcofm, ycofm, zcofm);

  CImg<> output_tiff(nz*2, nx/2+1, 1, 1, 0.f);
  if (b3Dout) {
    output_tiff.assign(floatimage, nx2, ny, nz, 1, true);
    if (interpkr>1)
      fixorigin(output_tiff, interpkr);
    rescale(output_tiff);
  }
  else {
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
        return false;
      }

    rescale(avg_output, nx, nz);
  }
  /* For side bands, combine bandre's and bandim's into bandplus */
  /* Shouldn't this be done later on the averaged bands? */

  output_tiff.save_tiff(ofiles.c_str());
}

/*  locate peak pixel to subpixel accuracy by fitting parabolas  */
void determine_center_and_background(float *stack3d, int nx, int ny, int nz, float *xc, float *yc, float *zc, float *background)
{
  int i, j, k, maxi=0, maxj=0, maxk=0, ind, nxy2, infocus_sec;
  int iminus, iplus, jminus, jplus, kminus, kplus;
  float maxval, reval, valminus, valplus;
  double sum;

  printf("In determine_center_and_background()\n");
  int nx2 = (nx/2+1)*2;   // in case nx is odd numbered, nx2 is nx+1, not nx+2
  nxy2 = nx2*ny;

  /* Search for the peak pixel */
  /* Be aware that stack3d is of dimension (nx+2)xnyxnz */
  maxval=0.0;
  for(k=0;k<nz;k++)
    for(i=0;i<ny;i++)
      for(j=0;j<nx;j++) {
        ind=k*nxy2+i*nx2+j;
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

  valminus = stack3d[kminus*nxy2+maxi*nx2+maxj];
  valplus  = stack3d[kplus *nxy2+maxi*nx2+maxj];
  *zc = maxk + fitparabola(valminus, maxval, valplus);

  *zc += 0.6;

  valminus = stack3d[maxk*nxy2+iminus*nx2+maxj];
  valplus  = stack3d[maxk*nxy2+iplus *nx2+maxj];
  *yc = maxi + fitparabola(valminus, maxval, valplus);

  valminus = stack3d[maxk*nxy2+maxi*nx2+jminus];
  valplus  = stack3d[maxk*nxy2+maxi*nx2+jplus];
  *xc = maxj + fitparabola(valminus, maxval, valplus);

  sum = 0;
  infocus_sec = floor(*zc);
  for (i=0; i<*yc-20; i++)
    for (j=0; j<nx; j++)
      sum += stack3d[infocus_sec*nxy2 + i*nx2 + j];
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

void fixorigin(CImg<> &otf3d, int kx2)
{
  // Project 3D OTF onto kx-ky plane for the +/- 45-degree radial lines
  int ny = otf3d.height();
  int nx = otf3d.width()/2; // real and imaginary

  CImg<> projected(nx*2, 1, 1, 1, 0.);

  for (int z=0; z<otf3d.depth(); z++)
    for (int x=1; x<nx; x++) {
      int xst = x<<1;
      // +45 degree line:
      projected(xst)   += otf3d(xst, x, z);
      projected(xst+1) += otf3d(xst+1, x, z);
      // -45 degree line:
      projected(xst)   += otf3d(xst, ny-x, z);
      projected(xst+1) += otf3d(xst+1, ny-x, z);
    }

  //
  std::complex<double> mean_val=0;
  std::complex<double> slope_numerator=0;
  std::complex<double> slope_denominat=0;
  std::complex<float> * projectedC = (std::complex<float> *) projected.data();

  for (int x=1; x<=kx2; x++)
    mean_val += projectedC[x];
  mean_val /= kx2;

  double mean_kx = (kx2+1)/2.; // the mean of [1, 2, ..., n] is (n+1)/2
  for (int x=1; x<=kx2; x++) {
    std::complex<double> complexval = projectedC[x];
    slope_numerator += (x - mean_kx) * (complexval - mean_val);
    slope_denominat += (x - mean_kx) * (x - mean_kx);
  }
  std::complex<double> slope = slope_numerator / slope_denominat;
  std::complex<double> valOrigin = mean_val - slope * mean_kx;   // intercept at kx=0
  std::cout << "otf3d(0) = " << otf3d(0, 0, 0) << "+i" << otf3d(1,0,0) << std::endl;
  otf3d(0, 0, 0) = valOrigin.real();
  otf3d(1, 0, 0) = valOrigin.imag();
  std::cout << "otf3d(0) = " << otf3d(0, 0, 0) << "+i" << otf3d(1,0,0) << std::endl;
}

bool fixorigin(std::complex<float> *otfkxkz, int nx, int nz, int kx2)
{
  // linear fit the value at kx=0 using kx in [1, kx2]
  double mean_kx = (kx2+1)/2.; // the mean of [1, 2, ..., n] is (n+1)/2

  printf("In fixorigin(), kx2=%d\n", kx2);

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

void rescale(CImg<> &otf3d)
{
  std::complex<float> * otf3dC = (std::complex<float> *) otf3d.data();
  size_t nxyz = otf3d.size() / 2;
  float valmax = 0., mag;
  for (unsigned i=0; i<nxyz; i++) {
    mag = std::abs(otf3dC[i]);
    if (mag > valmax)
      valmax = mag;
  }

  otf3d /= valmax;
}
