#include "linearDecon.h"

int main(int argc, char *argv[])
// argv[1] is the input file name; argv[2] is the OTF file name
{
  TIFFSetWarningHandler(NULL);
  CImg<unsigned short> raw(argv[1]);


  RL_interface_init(raw.width(), raw.height(), raw.depth(),
                    .104, .4, .104, .1, -32.8, -32.8, 0, argv[2]);

  CImg<float> result(get_output_nx(), get_output_ny(), get_output_nz());

  printf("result dimensions: %d, %d, %d\n", result.width(), result.height(), result.depth());
  RL_interface(raw.data(), raw.width(), raw.height(), raw.depth(), result.data(), 90., 10, 0);

  result.save("debug.tif");
  return 0; 
}
