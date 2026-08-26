/*
 * probe.c - the oracle harness for oracle-captures/foreign-nifti (issue #641).
 *
 * libvips has no NIfTI support in this build, so there is no `vips niftiload`
 * to shell out to the way every other capture area does. This links
 * nifti_clib's own libnifti2.a instead and asks the reference implementation
 * the questions a port has to answer: what a datatype code means, how the
 * version and the byte order are decided, where every header field sits, what
 * `nifti_image_read` does with a broken file, and what bytes end up in memory
 * once the loader has finished with them.
 *
 * Every number this prints comes out of a nifti_clib function call. Nothing
 * here re-implements the format.
 *
 * Built and driven by capture.py; not meant to be run by hand.
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "nifti2_io.h"

/* nifti2_io.h defines these inside its private NIFTI2_IO_C section, so they
 * are not visible to a caller. Repeated here with the values that section
 * gives them (nifti2_io.h, "#define LSB_FIRST 1" / "#define MSB_FIRST 2")
 * purely to LABEL what nifti_short_order() and nim->byteorder return; every
 * number reported is still the library's own. */
#define P_LSB_FIRST 1
#define P_MSB_FIRST 2

/* ------------------------------------------------------------------ *
 * A very small JSON writer. Enough for flat objects and arrays.
 * ------------------------------------------------------------------ */
static int comma_pending = 0;

static void sep(void) { if (comma_pending) printf(","); comma_pending = 1; }
static void obj_open(void) { sep(); printf("{"); comma_pending = 0; }
static void obj_close(void) { printf("}"); comma_pending = 1; }
static void arr_open(void) { sep(); printf("["); comma_pending = 0; }
static void arr_close(void) { printf("]"); comma_pending = 1; }
static void key(const char *k) { sep(); printf("\"%s\":", k); comma_pending = 0; }

static void jstr_raw(const char *s)
{
   printf("\"");
   if (s) {
      for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
         if (*p == '"' || *p == '\\') printf("\\%c", *p);
         else if (*p == '\n') printf("\\n");
         else if (*p == '\r') printf("\\r");
         else if (*p < 0x20 || *p >= 0x7f) printf("\\u%04x", *p);
         else printf("%c", *p);
      }
   }
   printf("\"");
   comma_pending = 1;
}

static void kv_str(const char *k, const char *v)
{
   key(k);
   if (!v) { printf("null"); comma_pending = 1; } else jstr_raw(v);
}
static void kv_int(const char *k, long long v) { key(k); printf("%lld", v); comma_pending = 1; }
static void kv_bool(const char *k, int v) { key(k); printf(v ? "true" : "false"); comma_pending = 1; }

/* JSON has no infinity or NaN, so those become strings and the caller can
 * see exactly which value the library produced. */
static void jnum(double v)
{
   if (v != v)                 { printf("\"NaN\""); }
   else if (v > 1.7e308)       { printf("\"Infinity\""); }
   else if (v < -1.7e308)      { printf("\"-Infinity\""); }
   else if (v == (long long)v && v > -1e15 && v < 1e15) printf("%lld", (long long)v);
   else printf("%.17g", v);
   comma_pending = 1;
}
static void kv_num(const char *k, double v) { key(k); jnum(v); }

static void kv_i64arr(const char *k, const int64_t *a, int n)
{
   key(k); arr_open();
   for (int i = 0; i < n; i++) { sep(); printf("%" PRId64, a[i]); comma_pending = 1; }
   arr_close();
}
static void kv_numarr(const char *k, const double *a, int n)
{
   key(k); arr_open();
   for (int i = 0; i < n; i++) { sep(); jnum(a[i]); }
   arr_close();
}

/* A fixed-width char field, exactly as it sits on disk: every byte, including
 * the NULs, so a port can see the padding convention rather than guess it. */
static void kv_bytes(const char *k, const void *p, int n)
{
   const unsigned char *b = (const unsigned char *)p;
   key(k); arr_open();
   for (int i = 0; i < n; i++) { sep(); printf("%d", b[i]); comma_pending = 1; }
   arr_close();
}

static void kv_hex(const char *k, const void *p, size_t n)
{
   const unsigned char *b = (const unsigned char *)p;
   char *buf = malloc(n * 2 + 1);
   for (size_t i = 0; i < n; i++) sprintf(buf + i * 2, "%02x", b[i]);
   buf[n * 2] = '\0';
   kv_str(k, buf);
   free(buf);
}

/* ------------------------------------------------------------------ *
 * env: what this build of the library is, and what the host is.
 * ------------------------------------------------------------------ */
static int cmd_env(void)
{
   obj_open();
   kv_int("sizeof_nifti_1_header", (long long)sizeof(nifti_1_header));
   kv_int("sizeof_nifti_2_header", (long long)sizeof(nifti_2_header));
   kv_int("sizeof_nifti_analyze75", (long long)sizeof(nifti_analyze75));
   kv_int("nifti_short_order", nifti_short_order());
   kv_str("nifti_short_order_meaning",
          nifti_short_order() == P_LSB_FIRST ? "LSB_FIRST (1)" : "MSB_FIRST (2)");
   kv_int("nifti_compiled_with_zlib", nifti_compiled_with_zlib());
   kv_int("nifti_test_datatype_sizes", nifti_test_datatype_sizes(0));
   /* The datatype table hands out nbyper 16 for DT_FLOAT128 and 32 for
    * DT_COMPLEX256, but those are only right where `long double` is 16
    * bytes wide. It is not on every target, and nifti_test_datatype_sizes
    * checks the table against ITSELF rather than against sizeof, so it
    * cannot catch the difference. These are the widths on this host. */
   kv_int("sizeof_float", (long long)sizeof(float));
   kv_int("sizeof_double", (long long)sizeof(double));
   kv_int("sizeof_long_double", (long long)sizeof(long double));
   obj_close();
   return 0;
}

/* ------------------------------------------------------------------ *
 * datatypes: sweep every code the library could be asked about.
 * ------------------------------------------------------------------ */
static int cmd_datatypes(void)
{
   /* Every code named in nifti1.h, plus a handful of codes that are NOT
    * named, so the refusal side is measured too rather than assumed. */
   static const int codes[] = {
      -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
      127, 128, 129, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1023, 1024,
      1025, 1279, 1280, 1281, 1535, 1536, 1537, 1791, 1792, 1793, 2047, 2048,
      2049, 2303, 2304, 2305, 4096, 65535
   };
   arr_open();
   for (size_t i = 0; i < sizeof(codes) / sizeof(codes[0]); i++) {
      int dt = codes[i], nbyper = -999, swapsize = -999;
      nifti_datatype_sizes(dt, &nbyper, &swapsize);
      obj_open();
      kv_int("code", dt);
      kv_int("nbyper", nbyper);
      kv_int("swapsize", swapsize);
      kv_int("bitpix", 8LL * nbyper);
      kv_str("nifti_datatype_string", nifti_datatype_string(dt));
      kv_str("nifti_datatype_to_string", nifti_datatype_to_string(dt));
      kv_int("nifti_is_inttype", nifti_is_inttype(dt));
      kv_int("nifti_is_valid_datatype", nifti_is_valid_datatype(dt));
      kv_int("valid_for_analyze", nifti_datatype_is_valid(dt, 0));
      kv_int("valid_for_nifti", nifti_datatype_is_valid(dt, 1));
      /* Round-trip the name back through the parser, which is how a port
       * would be asked to accept a textual datatype. */
      kv_int("from_string_of_to_string",
             nifti_datatype_from_string(nifti_datatype_to_string(dt)));
      obj_close();
   }
   arr_close();
   return 0;
}

/* ------------------------------------------------------------------ *
 * offsets: the on-disk layout of all three headers.
 * ------------------------------------------------------------------ */
#define OFF1(f) do { obj_open(); kv_str("field", #f); \
   kv_int("offset", (long long)offsetof(nifti_1_header, f)); \
   kv_int("size", (long long)sizeof(((nifti_1_header *)0)->f)); obj_close(); } while (0)
#define OFF2(f) do { obj_open(); kv_str("field", #f); \
   kv_int("offset", (long long)offsetof(nifti_2_header, f)); \
   kv_int("size", (long long)sizeof(((nifti_2_header *)0)->f)); obj_close(); } while (0)
#define OFFA(f) do { obj_open(); kv_str("field", #f); \
   kv_int("offset", (long long)offsetof(nifti_analyze75, f)); \
   kv_int("size", (long long)sizeof(((nifti_analyze75 *)0)->f)); obj_close(); } while (0)

static int cmd_offsets(void)
{
   obj_open();

   key("nifti_1_header"); arr_open();
   OFF1(sizeof_hdr); OFF1(data_type); OFF1(db_name); OFF1(extents);
   OFF1(session_error); OFF1(regular); OFF1(dim_info); OFF1(dim);
   OFF1(intent_p1); OFF1(intent_p2); OFF1(intent_p3); OFF1(intent_code);
   OFF1(datatype); OFF1(bitpix); OFF1(slice_start); OFF1(pixdim);
   OFF1(vox_offset); OFF1(scl_slope); OFF1(scl_inter); OFF1(slice_end);
   OFF1(slice_code); OFF1(xyzt_units); OFF1(cal_max); OFF1(cal_min);
   OFF1(slice_duration); OFF1(toffset); OFF1(glmax); OFF1(glmin);
   OFF1(descrip); OFF1(aux_file); OFF1(qform_code); OFF1(sform_code);
   OFF1(quatern_b); OFF1(quatern_c); OFF1(quatern_d); OFF1(qoffset_x);
   OFF1(qoffset_y); OFF1(qoffset_z); OFF1(srow_x); OFF1(srow_y);
   OFF1(srow_z); OFF1(intent_name); OFF1(magic);
   arr_close();

   key("nifti_2_header"); arr_open();
   OFF2(sizeof_hdr); OFF2(magic); OFF2(datatype); OFF2(bitpix); OFF2(dim);
   OFF2(intent_p1); OFF2(intent_p2); OFF2(intent_p3); OFF2(pixdim);
   OFF2(vox_offset); OFF2(scl_slope); OFF2(scl_inter); OFF2(cal_max);
   OFF2(cal_min); OFF2(slice_duration); OFF2(toffset); OFF2(slice_start);
   OFF2(slice_end); OFF2(descrip); OFF2(aux_file); OFF2(qform_code);
   OFF2(sform_code); OFF2(quatern_b); OFF2(quatern_c); OFF2(quatern_d);
   OFF2(qoffset_x); OFF2(qoffset_y); OFF2(qoffset_z); OFF2(srow_x);
   OFF2(srow_y); OFF2(srow_z); OFF2(slice_code); OFF2(xyzt_units);
   OFF2(intent_code); OFF2(intent_name); OFF2(dim_info); OFF2(unused_str);
   arr_close();

   key("nifti_analyze75"); arr_open();
   OFFA(sizeof_hdr); OFFA(data_type); OFFA(db_name); OFFA(extents);
   OFFA(session_error); OFFA(regular); OFFA(hkey_un0); OFFA(dim);
   OFFA(unused8); OFFA(unused9); OFFA(unused10); OFFA(unused11);
   OFFA(unused12); OFFA(unused13); OFFA(unused14); OFFA(datatype);
   OFFA(bitpix); OFFA(dim_un0); OFFA(pixdim); OFFA(vox_offset);
   OFFA(funused1); OFFA(funused2); OFFA(funused3); OFFA(cal_max);
   OFFA(cal_min); OFFA(compressed); OFFA(verified); OFFA(glmax);
   OFFA(glmin); OFFA(descrip); OFFA(aux_file); OFFA(orient);
   OFFA(originator); OFFA(generated); OFFA(scannum); OFFA(patient_id);
   OFFA(exp_date); OFFA(exp_time); OFFA(hist_un0); OFFA(views);
   OFFA(vols_added); OFFA(start_field); OFFA(field_skip); OFFA(omax);
   OFFA(omin); OFFA(smax); OFFA(smin);
   arr_close();

   obj_close();
   return 0;
}

/* ------------------------------------------------------------------ *
 * hdrver <file>: what nifti_header_version says, at several buffer sizes.
 * This is the whole of the version-and-endianness sentinel.
 * ------------------------------------------------------------------ */
static int cmd_hdrver(const char *path)
{
   static const size_t sizes[] = { 0, 1, 100, 347, 348, 349, 539, 540, 541 };
   unsigned char buf[600];
   size_t got;
   FILE *fp = fopen(path, "rb");

   memset(buf, 0, sizeof(buf));
   if (!fp) { obj_open(); kv_str("error", "cannot open"); obj_close(); return 1; }
   got = fread(buf, 1, sizeof(buf), fp);
   fclose(fp);

   obj_open();
   kv_int("bytes_available", (long long)got);
   key("nifti_header_version"); arr_open();
   for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
      obj_open();
      kv_int("nbytes", (long long)sizes[i]);
      kv_int("result", nifti_header_version((char *)buf, sizes[i]));
      obj_close();
   }
   arr_close();
   kv_int("null_buffer_result", nifti_header_version(NULL, 348));
   obj_close();
   return 0;
}

/* ------------------------------------------------------------------ *
 * names <path>: the filename side. Which .hdr goes with which .img,
 * what counts as an extension, what a missing half does.
 * ------------------------------------------------------------------ */
static int cmd_names(const char *path)
{
   char *ext, *hdr, *img1, *img2, *base;

   obj_open();
   kv_str("path", path);
   ext = nifti_find_file_extension(path);
   kv_str("nifti_find_file_extension", ext);
   kv_int("nifti_is_gzfile", nifti_is_gzfile(path));
   kv_int("nifti_validfilename", nifti_validfilename(path));
   kv_int("nifti_is_complete_filename", nifti_is_complete_filename(path));
   base = nifti_makebasename(path);
   kv_str("nifti_makebasename", base);
   if (base) free(base);
   hdr = nifti_findhdrname(path);
   kv_str("nifti_findhdrname", hdr);
   if (hdr) free(hdr);
   /* NIFTI_FTYPE_NIFTI1_1 is the single-file form, _2 the pair; the answer
    * differs, so both are asked. */
   img1 = nifti_findimgname(path, NIFTI_FTYPE_NIFTI1_1);
   kv_str("nifti_findimgname_ftype1", img1);
   if (img1) free(img1);
   img2 = nifti_findimgname(path, NIFTI_FTYPE_NIFTI1_2);
   kv_str("nifti_findimgname_ftype2", img2);
   if (img2) free(img2);
   kv_int("is_nifti_file", is_nifti_file(path));
   kv_int("nifti_get_filesize", (long long)nifti_get_filesize(path));
   obj_close();
   return 0;
}

/* ------------------------------------------------------------------ *
 * read <file> [maxvox]: the whole load, reported.
 * ------------------------------------------------------------------ */
static void dump_nim(nifti_image *nim, long maxbytes)
{
   obj_open();
   kv_int("ndim", nim->ndim);
   kv_i64arr("dim", nim->dim, 8);
   kv_int("nx", nim->nx); kv_int("ny", nim->ny); kv_int("nz", nim->nz);
   kv_int("nt", nim->nt); kv_int("nu", nim->nu); kv_int("nv", nim->nv);
   kv_int("nw", nim->nw);
   kv_int("nvox", nim->nvox);
   kv_int("nbyper", nim->nbyper);
   kv_int("datatype", nim->datatype);
   kv_str("datatype_string", nifti_datatype_string(nim->datatype));
   kv_numarr("pixdim", nim->pixdim, 8);
   /* qfac is pixdim[0] on disk but a field of its own in nifti_image, and
    * nim->pixdim[0] is left at 0; a port that reads spacing out of
    * pixdim[0] gets nothing. */
   kv_num("qfac", nim->qfac);
   kv_num("scl_slope", nim->scl_slope);
   kv_num("scl_inter", nim->scl_inter);
   kv_num("cal_min", nim->cal_min);
   kv_num("cal_max", nim->cal_max);
   kv_int("qform_code", nim->qform_code);
   kv_int("sform_code", nim->sform_code);
   kv_int("intent_code", nim->intent_code);
   kv_int("xyz_units", nim->xyz_units);
   kv_str("xyz_units_string", nifti_units_string(nim->xyz_units));
   kv_int("time_units", nim->time_units);
   kv_int("nifti_type", nim->nifti_type);
   kv_int("iname_offset", nim->iname_offset);
   kv_int("swapsize", nim->swapsize);
   kv_int("byteorder", nim->byteorder);
   kv_str("byteorder_meaning",
          nim->byteorder == P_LSB_FIRST ? "LSB_FIRST" :
          nim->byteorder == P_MSB_FIRST ? "MSB_FIRST" : "other");
   kv_int("byteorder_differs_from_host", nim->byteorder != nifti_short_order());
   kv_int("num_ext", nim->num_ext);
   kv_str("descrip", nim->descrip);
   kv_str("intent_name", nim->intent_name);
   kv_int("nifti_get_volsize", (long long)nifti_get_volsize(nim));
   kv_int("nifti_nim_has_valid_dims", nifti_nim_has_valid_dims(nim, 0));
   kv_int("nifti_nim_is_valid", nifti_nim_is_valid(nim, 0));
   kv_int("analyze75_orient", (int)nim->analyze75_orient);
   /* The filenames the loader settled on. Basename only: the absolute path
    * would make this capture machine-specific. */
   {
      const char *f = nim->fname, *n = nim->iname, *p;
      if (f && (p = strrchr(f, '/'))) f = p + 1;
      if (n && (p = strrchr(n, '/'))) n = p + 1;
      kv_str("fname", f);
      kv_str("iname", n);
   }
   /* nifti_image_load byte-swaps in place, so these are the bytes AFTER the
    * loader has finished: on a little-endian host, a big-endian file's data
    * comes back already swapped. */
   if (nim->data) {
      long n = (long)(nim->nvox * nim->nbyper);
      kv_int("data_bytes", n);
      if (maxbytes > 0 && n > maxbytes) n = maxbytes;
      kv_hex("data_hex_after_load", nim->data, (size_t)n);
   } else {
      kv_str("data_hex_after_load", NULL);
   }
   obj_close();
}

static int cmd_read(const char *path, long maxbytes)
{
   nifti_image *nim = nifti_image_read(path, 1);
   obj_open();
   if (!nim) {
      kv_bool("loaded", 0);
   } else {
      kv_bool("loaded", 1);
      key("nim"); dump_nim(nim, maxbytes);
      nifti_image_free(nim);
   }
   obj_close();
   return 0;
}

/* header-only read, which is the path that decides whether a file is
 * accepted at all; the pixels are a separate failure surface. */
static int cmd_readhdr(const char *path, int check)
{
   int nver = -99;
   void *hdr = nifti_read_header(path, &nver, check);
   obj_open();
   kv_int("check", check);
   kv_bool("got_header", hdr != NULL);
   kv_int("nver", nver);
   if (hdr) {
      if (nver == 2) {
         nifti_2_header *h = (nifti_2_header *)hdr;
         kv_int("sizeof_hdr", h->sizeof_hdr);
         kv_bytes("magic", h->magic, 8);
         kv_int("datatype", h->datatype);
         kv_int("bitpix", h->bitpix);
         kv_i64arr("dim", h->dim, 8);
         kv_num("vox_offset", (double)h->vox_offset);
         kv_num("scl_slope", h->scl_slope);
         kv_num("scl_inter", h->scl_inter);
         kv_int("nifti_hdr2_looks_good", nifti_hdr2_looks_good(h));
      } else {
         nifti_1_header *h = (nifti_1_header *)hdr;
         kv_int("sizeof_hdr", h->sizeof_hdr);
         kv_bytes("magic", h->magic, 4);
         kv_int("datatype", h->datatype);
         kv_int("bitpix", h->bitpix);
         key("dim"); arr_open();
         for (int i = 0; i < 8; i++) { sep(); printf("%d", h->dim[i]); comma_pending = 1; }
         arr_close();
         kv_num("vox_offset", h->vox_offset);
         kv_num("scl_slope", h->scl_slope);
         kv_num("scl_inter", h->scl_inter);
         kv_int("nifti_hdr1_looks_good", h ? nifti_hdr1_looks_good(h) : 0);
      }
      free(hdr);
   }
   obj_close();
   return 0;
}

/* ------------------------------------------------------------------ *
 * make: write a dataset through nifti_image_write, so the encode side
 * is the oracle's too. The payload is a deterministic byte ramp, which
 * makes the datatype interpretation visible in the readback.
 * ------------------------------------------------------------------ */
static int cmd_make(char **argv)
{
   /* make <prefix> <ni_ver> <ftype> <datatype> <nd> d1..d7 <slope> <inter> */
   const char *prefix = argv[2];
   int ni_ver = atoi(argv[3]);
   int ftype = atoi(argv[4]);
   int datatype = atoi(argv[5]);
   int64_t dims[8];
   double slope, inter;
   nifti_image *nim;
   unsigned char *p;
   int64_t nbytes;

   dims[0] = atoi(argv[6]);
   for (int i = 1; i <= 7; i++) dims[i] = atoll(argv[6 + i]);
   slope = atof(argv[14]);
   inter = atof(argv[15]);

   nim = nifti_make_new_nim(dims, datatype, 1);
   if (!nim) { obj_open(); kv_bool("made", 0); obj_close(); return 1; }

   /* nifti_set_filenames decides one file or two from nim->nifti_type, so
    * the type has to be set first; set_type=0 then stops it being reset
    * from the extension, which cannot tell NIfTI-1 from NIfTI-2. */
   nim->nifti_type = ftype;
   nim->scl_slope = slope;
   nim->scl_inter = inter;
   nifti_set_filenames(nim, prefix, 0, 0);
   nifti_set_iname_offset(nim, ni_ver);

   /* A ramp of consecutive bytes starting at 0x80. Whatever the datatype,
    * the same bytes land on disk, so the readback shows exactly how the code
    * reinterprets them. Starting at 0x80 rather than 0 matters: byte i is
    * (0x80 + i) & 0xff, so for the first 128 bytes the top bit is set and
    * the most significant byte of a value of ANY width carries a sign bit,
    * which makes a signed / unsigned confusion show up in the very first
    * number rather than hiding behind small positive values. */
   nbytes = nim->nvox * nim->nbyper;
   p = (unsigned char *)nim->data;
   for (int64_t i = 0; i < nbytes; i++) p[i] = (unsigned char)(0x80 + i);

   obj_open();
   kv_bool("made", 1);
   kv_int("write_status", nifti_image_write_status(nim));
   kv_int("nvox", nim->nvox);
   kv_int("nbyper", nim->nbyper);
   kv_int("iname_offset", nim->iname_offset);
   kv_int("nifti_type", nim->nifti_type);
   kv_hex("payload_written_hex", nim->data, (size_t)nbytes);
   obj_close();
   nifti_image_free(nim);
   return 0;
}

/* ------------------------------------------------------------------ *
 * swap <file>: run the three swap entry points over a header buffer and
 * report what each produces, since a port has to pick one.
 * ------------------------------------------------------------------ */
static int cmd_swap(const char *path)
{
   unsigned char raw[540], buf[540];
   FILE *fp = fopen(path, "rb");
   size_t got;

   memset(raw, 0, sizeof(raw));
   if (!fp) { obj_open(); kv_str("error", "cannot open"); obj_close(); return 1; }
   got = fread(raw, 1, sizeof(raw), fp);
   fclose(fp);

   obj_open();
   kv_int("bytes_read", (long long)got);
   kv_hex("original_first_60", raw, 60);

   memcpy(buf, raw, sizeof(buf));
   nifti_swap_as_nifti1((nifti_1_header *)buf);
   kv_hex("nifti_swap_as_nifti1_first_60", buf, 60);

   memcpy(buf, raw, sizeof(buf));
   nifti_swap_as_analyze((nifti_analyze75 *)buf);
   kv_hex("nifti_swap_as_analyze_first_60", buf, 60);

   memcpy(buf, raw, sizeof(buf));
   old_swap_nifti_header((nifti_1_header *)buf, 1);
   kv_hex("old_swap_nifti_header_is_nifti_first_60", buf, 60);

   memcpy(buf, raw, sizeof(buf));
   old_swap_nifti_header((nifti_1_header *)buf, 0);
   kv_hex("old_swap_nifti_header_not_nifti_first_60", buf, 60);

   memcpy(buf, raw, sizeof(buf));
   nifti_swap_as_nifti2((nifti_2_header *)buf);
   kv_hex("nifti_swap_as_nifti2_first_60", buf, 60);
   obj_close();
   return 0;
}

/* ------------------------------------------------------------------ *
 * swapfile <in> <out> <ni_ver> <hdr_bytes> <data_off> <swapsize>:
 * rewrite a file into the opposite byte order using the library's own
 * swap routines, so the big-endian fixtures are produced by the oracle
 * rather than by a hand-rolled struct packer.
 * ------------------------------------------------------------------ */
static int cmd_swapfile(char **argv)
{
   const char *in = argv[2], *out = argv[3];
   int ni_ver = atoi(argv[4]);
   long data_off = atol(argv[5]);
   int swapsize = atoi(argv[6]);
   unsigned char *buf;
   long len;
   FILE *fp = fopen(in, "rb");

   if (!fp) { obj_open(); kv_str("error", "cannot open input"); obj_close(); return 1; }
   fseek(fp, 0, SEEK_END); len = ftell(fp); fseek(fp, 0, SEEK_SET);
   buf = malloc((size_t)len);
   if (fread(buf, 1, (size_t)len, fp) != (size_t)len) {
      fclose(fp); free(buf);
      obj_open(); kv_str("error", "short read"); obj_close(); return 1;
   }
   fclose(fp);

   if (ni_ver == 2) nifti_swap_as_nifti2((nifti_2_header *)buf);
   else             nifti_swap_as_nifti1((nifti_1_header *)buf);

   if (swapsize > 1 && data_off < len)
      nifti_swap_Nbytes((len - data_off) / swapsize, swapsize, buf + data_off);

   fp = fopen(out, "wb");
   fwrite(buf, 1, (size_t)len, fp);
   fclose(fp);

   obj_open();
   kv_int("bytes", len);
   kv_int("data_offset", data_off);
   kv_int("swapsize", swapsize);
   kv_hex("header_first_16_after_swap", buf, 16);
   obj_close();
   free(buf);
   return 0;
}

int main(int argc, char **argv)
{
   int rv;

   if (argc < 2) { fprintf(stderr, "usage: probe <cmd> [args]\n"); return 2; }

   /* The library's own default debug level is 1 (nifti2_io.c, g_opts), which
    * is what prints the "** ERROR ..." lines the refusal records are made of,
    * so it is deliberately left alone. `--debug` raises it to 3 for the cases
    * where the quiet path needs explaining. */
   if (argc > 2 && !strcmp(argv[argc - 1], "--debug")) nifti_set_debug_level(3);

   if      (!strcmp(argv[1], "env"))       rv = cmd_env();
   else if (!strcmp(argv[1], "datatypes")) rv = cmd_datatypes();
   else if (!strcmp(argv[1], "offsets"))   rv = cmd_offsets();
   else if (!strcmp(argv[1], "hdrver"))    rv = cmd_hdrver(argv[2]);
   else if (!strcmp(argv[1], "names"))     rv = cmd_names(argv[2]);
   else if (!strcmp(argv[1], "read"))      rv = cmd_read(argv[2], argc > 3 ? atol(argv[3]) : 0);
   else if (!strcmp(argv[1], "readhdr"))   rv = cmd_readhdr(argv[2], argc > 3 ? atoi(argv[3]) : 1);
   else if (!strcmp(argv[1], "make"))      rv = cmd_make(argv);
   else if (!strcmp(argv[1], "swap"))      rv = cmd_swap(argv[2]);
   else if (!strcmp(argv[1], "swapfile"))  rv = cmd_swapfile(argv);
   else { fprintf(stderr, "unknown command %s\n", argv[1]); return 2; }

   printf("\n");
   return rv;
}
