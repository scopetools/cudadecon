

void PSF_to_OTF(char *psfFileName) {

  CImg<unsigned short> PSF(otffiles.c_str());
  float xcofm, ycofm, zcofm;
  float estBackground;
  determine_center_and_background(PSF.data(), PSF.width(), PSF.height(), PSF.depth(),
                                  &xcofm, &ycofm, &zcofm, &estBackground);
  printf("Center of mass is (%.3f, %.3f, %.3f)\n\n", xcofm, ycofm, zcofm);

  CImg<> psf_float(PSF);
  psf_float -= estBackground;

  rfftwnd_plan rfftplan3d = rfftw3d_create_plan(PSF.depth(), PSF.height(), PSF.width(),
                                                FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);

  printf("Before fft\n");
  CImg<> psfFFT(PSF.width()+2, PSF.height(), PSF.depth());
  rfftwnd_one_real_to_complex(rfftplan3d, psf_float.data(), (fftwf_complex *) psfFFT.data());

  fftwnd_destroy_plan(rfftplan3d);
  printf("After fft\n\n");

  /* modify the phase of bands, so that it corresponds to FFT of a bead at origin */
  printf("Shifting center...\n");
  fftwf_complex *bands = (fftwf_complex *) psfFFT.data();
  shift_center(bands, PSF.width(), PSF.height(), PSF.depth(), xcofm, ycofm, zcofm);

  CImg<> output_tiff(PSF.depth()*2, PSF.width()/2+1, 1, 1, 0.f);
  fftwf_complex * avg_output = (fftwf_complex *) output_tiff.data();

  radialft(bands, nx, ny, nz, avg_output);

  fixorigin(avg_output, nx, nz, interpkr[0], interpkr[1]);
  rescale(avg_output, nx, nz);
}
