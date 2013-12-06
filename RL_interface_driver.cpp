#define CUDADECON_IMPORT

#include "linearDecon.h"

int main(int argc, char *argv[])
// argv[1] is the input file name; argv[2] is the OTF file name
{
  TIFFSetWarningHandler(NULL);
  CImg<unsigned short> raw(argv[1]);

  // Step 1: call RL_interface_init() once for all images to be 
  // acquired at this session; see linearDecon.h for detailed parameter descriptions
  RL_interface_init(raw.width(), raw.height(), raw.depth(),
                    .104, .4, .104, .1, -32.8, -32.8, 0, argv[2]);

  // Step 2: allocate buffer for result image, whose dimension might be different than 
  // the raw because CUFFT prefers dimensions to be factorizable to 2, 3, 5, and 7
  // Here for convenience I used CImg to allocate the buffer; but it's not needed.
  CImg<float> result(get_output_nx(), get_output_ny(), get_output_nz());

  // Step 3: call the gut of RL deconvolution; see linearDecon.h for detailed parameter descriptions
  // raw.data() and result.data() are just the plain pointers to the memory underneath "raw" and "results" CImg objects
  RL_interface(raw.data(), raw.width(), raw.height(), raw.depth(), result.data(), 90., 10, 0);

  // Saving result into a file for debug; not essential.
  result.save("debug.tif");

  // Need to call this to release a GPU buffer before the whole program quits:
  RL_cleanup();

  return 0; 
}
