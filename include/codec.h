//struct Codec {
  /* encode */ /* bool (in, insize, out, outsize) */
  /* decode */ /* bool (in, insize, out, outsize) */
  /* new */    /* ??? */
  /* cleanup *//* bool () */
//}
//
typedef enum {
    Huffman,
    Golomb,
    Arithmetic,
    LZ77 = 256,
// ...
    BWT = 512,
    CTW,
    Delta,
    DMC,
    MTF,
    PPM
} codec_name_t;
