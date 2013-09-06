#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <vector>
#include <iostream>
#include <complex>

#include <stdio.h>

#include <IMInclude.h>
#include <sfftw.h>
#include <srfftw.h>

#include <tiffio.h>

#define cimg_use_tiff
#include <CImg.h>
using namespace cimg_library;

extern int save_tiff(TIFF *tif, const unsigned int directory, int colind, const int nwaves, int width, int height, float * buffer);
extern int load_tiff(TIFF *const tif, const unsigned int directory, const unsigned colind, float *const buffer);

float fitparabola(float a1, float a2, float a3);
void determine_center_and_background(float *stack5phases, int nx, int ny, int nz, float *xc, float *yc, float *zc, float *background);
void shift_center(std::complex<float> *bands, int nx, int ny, int nz, float xc, float yc, float zc);
void cleanup(std::complex<float> *otfkxkz, int nx, int nz, float dkr, float dkz, float linespacing, int lambdanm, int twolens, float NA, float NIMM);
void radialft(std::complex<float> *bands, int nx, int ny, int nz, std::complex<float> *avg);
void fixorigin(std::complex<float> *otfkxkz, int nx, int nz, int kx1, int kx2);
void rescale(std::complex<float> *otfkxkz, int nx, int nz);

void outputMRC(int ostream_no, std::complex<float> *bands, IW_MRC_HEADER *header, int nx, int ny, int nz, float dkr, float dkz);
void outputTIFF(TIFF *tif, std::complex<float> *bands, int nx, int ny, int nz, float dkr, float dkz);


int main(int argc, char **argv)
{
  std::string ifiles, ofiles, I2Mfiles, order0files;
  int istream_no=1, ostream_no=2;
  int ixyz[3], mxyz[3], pixeltype;      /* variables for IMRdHdr call */
  float min, max, mean;                 /* variables for IMRdHdr call */
  int nx, ny, nz, nxy;
  int i, j, z;
  float dr, dz, dkr, dkz, background, estBackground, xcofm, ycofm, zcofm;
  float *floatimage;
  std::complex<float> *bands, *avg_output;
  rfftwnd_plan rfftplan3d;
  IW_MRC_HEADER header, otfheader;
  std::vector<int> interpkr(2);
  float NA, NIMM;
  int lambdanm;
  int krmax=0;
  bool bDoCleanup = true;
  bool bUserBackground = false;

  po::options_description progopts;
  progopts.add_options()
    ("na", po::value<float>(&NA)->default_value(1.4), "NA of detection objective")
    ("nimm", po::value<float>(&NIMM)->default_value(1.515), "refractive index of immersion medium")
    ("xyres", po::value<float>(&dr)->default_value(.104), "x-y pixel size")
    ("zres", po::value<float>(&dz)->default_value(.104), "z pixel size")
    ("wavelength", po::value<int>(&lambdanm)->default_value(530), "emission wavelength in nm")
    ("fixorigin", po::value< std::vector<int> >(&interpkr)->multitoken(),
     "extrapolate near the origin using these pixels on Kr axis")
    ("krmax", po::value<int>(&krmax)->default_value(0),
     "pixels outside this limit will be zeroed (overwriting estimated value from NA and NIMM)")
    ("nocleanup", po::value<bool>(&bDoCleanup)->implicit_value(false), "elect not to do clean-up outside OTF support")
    ("background", po::value<float>(&background), "use user-supplied background instead of the estimated")
    ("input-file", po::value<std::string>(&ifiles), "input file")
    ("output-file", po::value<std::string>(&ofiles), "output file")
    ("help,h", "produce help message")
    ;

  po::positional_options_description p;
  p.add("input-file", 1);
  p.add("output-file", 1);

/* Parse commandline option */
  po::variables_map varsmap;

  store(po::command_line_parser(argc, argv).
        options(progopts).positional(p).run(), varsmap);
  notify(varsmap);

  if (varsmap.count("help")) {
    std::cout << progopts << "\n";
    return 0;
  }

  if (varsmap.count("background")) {
    bUserBackground = true;
  }


  // printf("NA=%f,fixorign=%d,%d\ndr=%f, dz=%f\n", NA, interpkr[0], interpkr[1], dr, dz);

  IMAlPrt(0);

  if (!ifiles.length()) {
    printf(" PSF dataset file name: ");
    std::cin >> ifiles;
  }
  if (!ofiles.length()) {
    printf(" Output OTF file name: ");
    std::cin >> ofiles;
  }

  TIFF *input_tiff = TIFFOpen(ifiles.c_str(), "r");

  CImg<> rawtiff;
  bool bUseTIFF = false;

  if (! input_tiff) {
    if (IMOpen(istream_no, ifiles.c_str(), "ro")) {
      fprintf(stderr, "File %s does not exist.\n", ifiles.c_str());
      return -1;
    }
    IMRdHdr(istream_no, ixyz, mxyz, &pixeltype, &min, &max, &mean);
    IMGetHdr(istream_no, &header);

    nx = header.nx;
    ny = header.ny;
    nz = header.nz;
    dr = header.ylen;
    dz = header.zlen;

    if (IMOpen(ostream_no, ofiles.c_str(), "new")) {
      fprintf(stderr, "File %s can not be created.\n", ofiles.c_str());
      return -1;
    }
  }
  else {
    bUseTIFF = true;
    TIFFSetWarningHandler(NULL);
    TIFFClose(input_tiff);

    rawtiff.assign(ifiles.c_str());
    nz = rawtiff.depth();
    ny = rawtiff.height();
    nx = rawtiff.width();
  }

  printf("nx=%d, ny=%d, nz=%d\n", nx, ny, nz);

  dkr = 1/(ny*dr);
  dkz = 1/(nz*dz);

  nxy=(nx+2)*ny;

  floatimage = (float *) malloc(nxy*nz*sizeof(float));
  bands = (std::complex<float> *) floatimage;

  printf("Reading data...\n\n");

  for(z=0; z<nz; z++)
    if (!bUseTIFF)
      for (i=0; i<ny; i++)
        IMRdLin(istream_no, floatimage+z*nxy+i*(nx+2));
    else {
      for (i=0; i<ny; i++) {
        for (j=0; j<nx; j++)
          floatimage[z*nxy+i*(nx+2)+j] = rawtiff(j, i, z);
      }
    }

  /* Before FFT, use center band to estimate bead center position */
  determine_center_and_background(floatimage, nx, ny, nz, &xcofm, &ycofm, &zcofm, &estBackground);

  printf("Center of mass is (%.3f, %.3f, %.3f)\n\n", xcofm, ycofm, zcofm);

  if (!bUserBackground)
    background = estBackground;
  printf("Background is %.3f\n", background);

  for(z=0; z<nz; z++) {
    for(i=0;i<ny;i++)
      for(j=0;j<nx;j++)
        floatimage[z*nxy + i*(nx+2) + j] -= background;
  }

  rfftplan3d = rfftw3d_create_plan(nz, ny, nx, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE);
  printf("Before fft\n");
  rfftwnd_one_real_to_complex(rfftplan3d, floatimage, NULL);

  fftwnd_destroy_plan(rfftplan3d);
  printf("After fft\n\n");

  /* modify the phase of bands, so that it corresponds to FFT of a bead at origin */
  printf("Shifting center...\n");
  shift_center(bands, nx, ny, nz, xcofm, ycofm, zcofm);

  CImg<> output_tiff(nz*2, nx/2+1, 1, 1, 0.f);
  avg_output = (std::complex<float> *) output_tiff.data();

  radialft(bands, nx, ny, nz, avg_output);

  if (!bUseTIFF)
    lambdanm = header.iwav1;

  if (bDoCleanup)
    cleanup(avg_output, nx, nz, dkr, dkz, 1.0, lambdanm, krmax, NA, NIMM);

  if (interpkr[0] > 0 || interpkr[1] > 0)
    fixorigin(avg_output, nx, nz, interpkr[0], interpkr[1]);
  rescale(avg_output, nx, nz);

  /* For side bands, combine bandre's and bandim's into bandplus */
  /* Shouldn't this be done later on the averaged bands? */

  if (!bUseTIFF) {
    otfheader = header;
    outputMRC(ostream_no, avg_output, &otfheader, nx, ny, nz, dkr, dkz);
  }
  else
    output_tiff.save_tiff(ofiles.c_str());
}

/*  locate peak pixel to subpixel accuracy by fitting parabolas  */
void determine_center_and_background(float *stack3d, int nx, int ny, int nz, float *xc, float *yc, float *zc, float *background)
{
  int i, j, k, maxi, maxj, maxk, ind, nxy2, infocus_sec;
  int iminus, iplus, jminus, jplus, kminus, kplus;
  float maxval, reval, valminus, valplus;
  double sum;

  printf("In determine_center_and_background()\n");
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
          // std::cout << avg_output[indout] << std::endl;
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

  printf("krmax=%f\n", krmax);
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


void outputMRC(int ostream_no, std::complex<float> *bands, IW_MRC_HEADER *header, int nx, int ny, int nz, float dkr, float dkz)
{

  printf("In outputMRC()\n");

  header->nx = nz;
  header->ny = nx/2+1;
  header->nz = 1;
  header->mode = IW_COMPLEX;
  header->xlen = dkz;
  header->ylen = dkr;
  header->zlen = 0;
  header->amin = 0;
  header->amax = 1;

  IMPutHdr(ostream_no, header);
  IMWrSec(ostream_no, bands);
  IMWrHdr(ostream_no, header->label, 0, header->amin, header->amax, header->amean);
  IMClose(ostream_no);
}

void outputTIFF(TIFF *tif, std::complex<float> *bands, int nx, int ny, int nz, float dkr, float dkz)
{
  TIFFSetField(tif, TIFFTAG_XRESOLUTION, dkz);
  TIFFSetField(tif, TIFFTAG_YRESOLUTION, dkr);
  save_tiff(tif, 0, 0, 1, nz*2, nx/2+1, (float*) bands);
}


void fixorigin(std::complex<float> *otfkxkz, int nx, int nz, int kx1, int kx2)
{
  float meani, slope, avg, lineval;
  int numvals, i, j;
  double *sum, totsum=0, ysum=0, sqsum=0;

  printf("In fixorigin()\n");
  meani = 0.5*(kx1+kx2);
  numvals = kx2-kx1+1;
  sum = (double *) malloc((kx2+1)*sizeof(double));

  for (i=0; i<=kx2; i++) {
    sum[i] = 0;

    for (j=0; j<nz; j++)
      sum[i] += otfkxkz[i*nz+j].real(); /* imaginary parts will cancel each other because of symmetry */
    if (i>=kx1) {
      totsum += sum[i];
      ysum += sum[i] * (i-meani);
      sqsum += (i-meani) * (i-meani);
    }
  }
  slope = ysum / sqsum;
  avg = totsum / numvals;

  for (i=0; i<kx1; i++) {
    lineval = avg + (i-meani)*slope;
    otfkxkz[i*nz] -= /*std::complex<float>*/(sum[i] - lineval);
  }

  free(sum);
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

