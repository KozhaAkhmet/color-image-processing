# color-image-processing

A university coursework project (YTU Mathematical Engineering, Image Processing) implementing color image histogram equalization. Custom C code reads/writes the raw PPM image format from scratch (`get_image_data`/`write_image`), and OpenCV's YCrCb-based histogram equalization (`equalizeIntensity`) is used to actually equalize the color image, per the assignment's suggested approach of converting color spaces (RGB ↔ YCbCr) before equalizing.

## Known Issue

`main()` currently calls `equalizeIntensity(picture)`, but `equalizeIntensity` expects a `cv::Mat` while `picture` is the custom `ppm_file` struct — this won't compile as-is. It also discards the `cv::Mat` that `equalizeIntensity` returns rather than assigning it back, so even with a fixed call, `write_image` would still write out the *original*, un-equalized pixel data. To actually apply the equalization, the raw `rdata`/`gdata`/`bdata` buffers from `ppm_file` need to be packed into a `cv::Mat` before the call, and the equalized result needs to be unpacked back into a `ppm_file` before `write_image`.

## Build

Requires OpenCV (`find_package(OpenCV REQUIRED)` in `CMakeLists.txt`):

```sh
mkdir build && cd build
cmake ..
make
./ColorImageProcess
```

Test image `mandrill.ppm` is included; output is written to `pnr.ppm`.
