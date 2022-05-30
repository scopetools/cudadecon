#include <iostream>
#include <complex>

#define cimg_use_tiff
#include <CImg.h>

using namespace cimg_library;



int main(int argc, char *argv[])
{
  TIFFSetWarningHandler(NULL);

  CImg<float> atiff(argv[1]);
  CImg<> btiff(atiff.width()/2, atiff.height(), atiff.depth());

  std::cout << btiff.width() << ", " << btiff.height() << ", " << btiff.depth() << std::endl;

  for (unsigned i=0; i< btiff.size(); i++)
    btiff(i) = sqrt(abs(std::complex<float> (atiff(2*i), atiff(2*i+1))));

  btiff.display(0, false);
}
