#include <tiffio.h>

#define cimg_for1(bound,i) for (int i = 0; i<(int)(bound); ++i)
#define cimg_forX(width,x) cimg_for1(width,x)
#define cimg_forY(height,y) cimg_for1(height,y)
#define cimg_forZ(depth,z) cimg_for1(depth,z)
#define cimg_forXY(width,height,x,y) cimg_forY(height,y) cimg_forX(width,x)


// "t" is the datatype of the TIFF file, while "T" is datatype of memory storage
template<typename t, typename T>
void _load_tiff_tiled_contig(TIFF *const tif, const uint16 samplesperpixel, const uint32 nx, const uint32 ny, const uint32 tw, const uint32 th, const int colind, T *const buffer) {
  t *const buf = (t*)_TIFFmalloc(TIFFTileSize(tif));
  if (buf) {
    for (unsigned int row = 0; row<ny; row+=th)
      for (unsigned int col = 0; col<nx; col+=tw) {
        if (TIFFReadTile(tif,buf,col,row,0,0)<0) {
          _TIFFfree(buf); TIFFClose(tif);
          // throw CImgException(_cimg_instance
          //                     "load_tiff() : Invalid tile in file '%s'.",
          //                     cimg_instance,
          //                     TIFFFileName(tif));
        }
        const t *ptr = buf;
        for (unsigned int rr = row; rr<(row+th) && rr<ny; ++rr)
          for (unsigned int cc = col; cc<(col+tw) && cc<nx; ++cc)
            //            for (unsigned int vv = 0; vv<samplesperpixel; ++vv)
            buffer[rr*nx+cc] = (T) ptr[(rr-row)*tw*samplesperpixel + (cc-col)*samplesperpixel + colind];
      }
    _TIFFfree(buf);
  }
}

template<typename t, typename T>
void _load_tiff_tiled_separate(TIFF *const tif, const uint16 samplesperpixel, const uint32 nx, const uint32 ny, const uint32 tw, const uint32 th, const int colind, T *const buffer) {
  t *const buf = (t*)_TIFFmalloc(TIFFTileSize(tif));
  if (buf) {
    //    for (unsigned int vv = 0; vv<samplesperpixel; ++vv)
      for (unsigned int row = 0; row<ny; row+=th)
        for (unsigned int col = 0; col<nx; col+=tw) {
          if (TIFFReadTile(tif,buf,col,row,0,colind)<0) {
            _TIFFfree(buf); TIFFClose(tif);
            // throw CImgException(_cimg_instance
            //                     "load_tiff() : Invalid tile in file '%s'.",
            //                     cimg_instance,
            //                     TIFFFileName(tif));
          }
          const t *ptr = buf;
          for (unsigned int rr = row; rr<(row+th) && rr<ny; ++rr)
            for (unsigned int cc = col; cc<(col+tw) && cc<nx; ++cc)
              buffer[rr*nx+cc] = (T) *(ptr++);
        }
    _TIFFfree(buf);
  }
}

template<typename t, typename T>
void _load_tiff_contig(TIFF *const tif, const uint16 samplesperpixel, const uint32 nx, const uint32 ny, const int colind, T *const buffer) {
  t *const buf = (t*)_TIFFmalloc(TIFFStripSize(tif));
  if (buf) {
    uint32 row, rowsperstrip = (uint32)-1;
    TIFFGetField(tif,TIFFTAG_ROWSPERSTRIP,&rowsperstrip);
    for (row = 0; row<ny; row+= rowsperstrip) {
      uint32 nrow = (row+rowsperstrip>ny?ny-row:rowsperstrip);
      tstrip_t strip = TIFFComputeStrip(tif, row, 0);
      if ((TIFFReadEncodedStrip(tif,strip,buf,-1))<0) {
        _TIFFfree(buf); TIFFClose(tif);
        // throw CImgException(_cimg_instance
        //                     "load_tiff() : Invalid strip in file '%s'.",
        //                     cimg_instance,
        //                     TIFFFileName(tif));
      }
      const t *ptr = buf;
      for (unsigned int rr = 0; rr<nrow; ++rr)
        for (unsigned int cc = 0; cc<nx; ++cc)
          for (int vv = 0; vv<samplesperpixel; ++vv)
            if (vv==colind) 
              buffer[(row+rr)*nx+cc] = (T) *(ptr++);
            else
              ptr++;
    }
    _TIFFfree(buf);
  }
}

template<typename t, typename T>
void _load_tiff_separate(TIFF *const tif, const uint16 samplesperpixel, const uint32 nx, const uint32 ny, const int colind, T *const buffer) {
  t *buf = (t*)_TIFFmalloc(TIFFStripSize(tif));
  if (buf) {
    uint32 row, rowsperstrip = (uint32)-1;
    TIFFGetField(tif,TIFFTAG_ROWSPERSTRIP,&rowsperstrip);
    // for (unsigned int vv = 0; vv<samplesperpixel; ++vv)
      for (row = 0; row<ny; row+= rowsperstrip) {
        uint32 nrow = (row+rowsperstrip>ny?ny-row:rowsperstrip);
        tstrip_t strip = TIFFComputeStrip(tif, row, colind);
        if ((TIFFReadEncodedStrip(tif,strip,buf,-1))<0) {
          _TIFFfree(buf); TIFFClose(tif);
          // throw CImgException(_cimg_instance
          //                     "load_tiff() : Invalid strip in file '%s'.",
          //                     cimg_instance,
          //                     TIFFFileName(tif));
        }
        const t *ptr = buf;
        for (unsigned int rr = 0;rr<nrow; ++rr)
          for (unsigned int cc = 0; cc<nx; ++cc)
            buffer[(row+rr)*nx+cc] = (T) *(ptr++);
      }
    _TIFFfree(buf);
  }
}

#ifdef __SIRECON_USE_TIFF__
extern "C" int load_tiff(TIFF *const tif, const unsigned int directory, const unsigned colind, float *const buffer);
extern "C" int save_tiff(TIFF *tif, const unsigned int directory, int colind, const int nwaves, int width, int height, float * buffer , int bIsComplex);

#endif


//template<typename T>
int load_tiff(TIFF *const tif, const unsigned int directory, const unsigned colind, float *const buffer) {
  //Read the current directory of channel colind into buffer 
  if (!TIFFSetDirectory(tif,directory)) return 0;
  uint16 samplesperpixel, bitspersample;
  uint16 sampleformat = SAMPLEFORMAT_UINT;
  uint32 nx,ny;
  // const char *const filename = TIFFFileName(tif);
  TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&nx);
  TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&ny);
  TIFFGetField(tif,TIFFTAG_SAMPLESPERPIXEL,&samplesperpixel);
  TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleformat);
  TIFFGetFieldDefaulted(tif,TIFFTAG_BITSPERSAMPLE,&bitspersample);
  //  assign(nx,ny,1,samplesperpixel);
  if (bitspersample!=8 || !(samplesperpixel==3 || samplesperpixel==4)) {
    uint16 photo, config;
    TIFFGetField(tif,TIFFTAG_PLANARCONFIG,&config);
    TIFFGetField(tif,TIFFTAG_PHOTOMETRIC,&photo);
    if (TIFFIsTiled(tif)) {
      uint32 tw, th;
      TIFFGetField(tif,TIFFTAG_TILEWIDTH,&tw);
      TIFFGetField(tif,TIFFTAG_TILELENGTH,&th);
      if (config==PLANARCONFIG_CONTIG) switch (bitspersample) {
        case 8 : {
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_tiled_contig<unsigned char, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          else _load_tiff_tiled_contig<signed char, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
        } break;
        case 16 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_tiled_contig<unsigned short, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          else _load_tiff_tiled_contig<short, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          break;
        case 32 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_tiled_contig<unsigned int, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          else if (sampleformat==SAMPLEFORMAT_INT) _load_tiff_tiled_contig<int, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          else _load_tiff_tiled_contig<float, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          break;
        } else switch (bitspersample) {
        case 8 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_tiled_separate<unsigned char, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          else _load_tiff_tiled_separate<signed char, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          break;
        case 16 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_tiled_separate<unsigned short, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          else _load_tiff_tiled_separate<short, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          break;
        case 32 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_tiled_separate<unsigned int, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          else if (sampleformat==SAMPLEFORMAT_INT) _load_tiff_tiled_separate<int, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          else _load_tiff_tiled_separate<float, float>(tif,samplesperpixel,nx,ny,tw,th, colind, buffer);
          break;
        }
    } else {
      if (config==PLANARCONFIG_CONTIG) switch (bitspersample) {
        case 8 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_contig<unsigned char, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          else _load_tiff_contig<signed char, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          break;
        case 16 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_contig<unsigned short, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          else _load_tiff_contig<short, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          break;
        case 32 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_contig<unsigned int, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          else if (sampleformat==SAMPLEFORMAT_INT) _load_tiff_contig<int, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          else _load_tiff_contig<float, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          break;
        } else switch (bitspersample){
        case 8 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_separate<unsigned char, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          else _load_tiff_separate<signed char, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          break;
        case 16 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_separate<unsigned short, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          else _load_tiff_separate<short, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          break;
        case 32 :
          if (sampleformat==SAMPLEFORMAT_UINT) _load_tiff_separate<unsigned int, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          else if (sampleformat==SAMPLEFORMAT_INT) _load_tiff_separate<int, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          else _load_tiff_separate<float, float>(tif,samplesperpixel,nx,ny, colind, buffer);
          break;
        }
    }
  } else {
    uint32 *const raster = (uint32*)_TIFFmalloc(nx*ny*sizeof(uint32));
    if (!raster) {
      _TIFFfree(raster); TIFFClose(tif);
      // throw CImgException(_cimg_instance
      //                     "load_tiff() : Failed to allocate memory (%s) for file '%s'.",
      //                     cimg_instance,
      //                     cimg::strbuffersize(nx*ny*sizeof(uint32)),filename);
    }
    TIFFReadRGBAImage(tif,nx,ny,raster,0);
    switch (samplesperpixel) {
    case 1 : {
      cimg_forXY(nx,ny,x,y) buffer[nx*y+x] = (float)((raster[nx*(ny-1-y)+x] + 128)/257);
    } break;
    case 3 : {
      cimg_forXY(nx,ny,x,y) {
        int ind=y*nx+x, nxy=nx*ny;
        buffer[ind] = (float)TIFFGetR(raster[nx*(ny-1-y)+x]);
        buffer[nxy+ind] = (float)TIFFGetG(raster[nx*(ny-1-y)+x]);
        buffer[2*nxy+ind] = (float)TIFFGetB(raster[nx*(ny-1-y)+x]);
      }
    } break;
    case 4 : {
      cimg_forXY(nx,ny,x,y) {
        int ind=y*nx+x, nxy=nx*ny;
        buffer[ind] = (float)TIFFGetR(raster[nx*(ny-1-y)+x]);
        buffer[nxy+ind] = (float)TIFFGetG(raster[nx*(ny-1-y)+x]);
        buffer[2*nxy+ind] = (float)TIFFGetB(raster[nx*(ny-1-y)+x]);
        buffer[3*nxy+ind] = (float)TIFFGetA(raster[nx*(ny-1-y)+x]);
      }
    } break;
    }
    _TIFFfree(raster);
  }
  return 1;
}


int save_tiff(TIFF *tif, const unsigned int directory, int colind, /*const t& pixel_t,*/ const int nwaves, // const unsigned int compression,
              int width, int height, float * buffer) {
  // In case of saving complex-number file (bIsComplex==1), colind indicates saving real or imag part

  if (!tif) return 0;
  // const char *const filename = TIFFFileName(tif);
  uint32 rowsperstrip = (uint32)-1;
  uint16 spp = nwaves, bpp = 32, photometric;
  // if (spp==3 || spp==4) photometric = PHOTOMETRIC_RGB;
  /*else */photometric = PHOTOMETRIC_MINISBLACK;
  TIFFSetDirectory(tif,directory);
  TIFFSetField(tif,TIFFTAG_IMAGEWIDTH,width);
  TIFFSetField(tif,TIFFTAG_IMAGELENGTH,height);
  TIFFSetField(tif,TIFFTAG_ORIENTATION,ORIENTATION_TOPLEFT);
  TIFFSetField(tif,TIFFTAG_SAMPLESPERPIXEL,spp);
  // if (cimg::type<t>::is_float())
    TIFFSetField(tif,TIFFTAG_SAMPLEFORMAT,3);
  // else if (cimg::type<t>::min()==0) TIFFSetField(tif,TIFFTAG_SAMPLEFORMAT,1);
  // else TIFFSetField(tif,TIFFTAG_SAMPLEFORMAT,2);
  TIFFSetField(tif,TIFFTAG_BITSPERSAMPLE,bpp);
  TIFFSetField(tif,TIFFTAG_PLANARCONFIG,PLANARCONFIG_SEPARATE);
  TIFFSetField(tif,TIFFTAG_PHOTOMETRIC,photometric);
  // TIFFSetField(tif,TIFFTAG_COMPRESSION,compression?(compression-1):COMPRESSION_NONE);
  rowsperstrip = TIFFDefaultStripSize(tif,rowsperstrip);
  TIFFSetField(tif,TIFFTAG_ROWSPERSTRIP,rowsperstrip);
  TIFFSetField(tif,TIFFTAG_FILLORDER,FILLORDER_MSB2LSB);
  // t *const buf = (t*)_TIFFmalloc(TIFFStripSize(tif));
  float *const buf = (float*)_TIFFmalloc(TIFFStripSize(tif));
  if (buf) {
    for (unsigned int row = 0; row<height; row+=rowsperstrip) {
      uint32 nrow = (row + rowsperstrip>height?height-row:rowsperstrip);
      tstrip_t strip = TIFFComputeStrip(tif,row,colind);
      tsize_t i = 0;
      for (unsigned int rr = 0; rr<nrow; ++rr)
        for (unsigned int cc = 0; cc<width; ++cc)
          buf[i++] = buffer[cc+(row + rr)*width];
      if (TIFFWriteEncodedStrip(tif,strip,buf,i*sizeof(float))<0)
        return 0;
        // throw CImgException(_cimg_instance
        //                     "save_tiff() : Invalid strip writting when saving file '%s'.",
        //                     cimg_instance,
        //                     filename?filename:"(FILE*)");
    }
    _TIFFfree(buf);
  }
  TIFFWriteDirectory(tif);
  return 1;
}
