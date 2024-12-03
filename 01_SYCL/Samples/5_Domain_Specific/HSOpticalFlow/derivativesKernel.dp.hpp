/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  stride  image stride
/// \param[out] Ix      x derivative
/// \param[out] Iy      y derivative
/// \param[out] Iz      temporal derivative
///////////////////////////////////////////////////////////////////////////////
void ComputeDerivativesKernel(
    int width, int height, int stride, float *Ix, float *Iy, float *Iz,
    sycl::ext::oneapi::experimental::sampled_image_handle texSource,
    sycl::ext::oneapi::experimental::sampled_image_handle texTarget,
    const sycl::nd_item<3> &item_ct1) {
  const int ix = item_ct1.get_local_id(2) +
                 item_ct1.get_group(2) * item_ct1.get_local_range(2);
  const int iy = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range(1);

  const int pos = ix + iy * stride;

  if (ix >= width || iy >= height) return;

  float dx = 1.0f / (float)width;
  float dy = 1.0f / (float)height;

  float x = ((float)ix + 0.5f) * dx;
  float y = ((float)iy + 0.5f) * dy;

  float t0, t1;
  // x derivative
  t0 = sycl::ext::oneapi::experimental::sample_image<float>(
      texSource, sycl::float2(x - 2.0f * dx, y));
  t0 -= sycl::ext::oneapi::experimental::sample_image<float>(
            texSource, sycl::float2(x - 1.0f * dx, y)) *
        8.0f;
  t0 += sycl::ext::oneapi::experimental::sample_image<float>(
            texSource, sycl::float2(x + 1.0f * dx, y)) *
        8.0f;
  t0 -= sycl::ext::oneapi::experimental::sample_image<float>(
      texSource, sycl::float2(x + 2.0f * dx, y));
  t0 /= 12.0f;

  t1 = sycl::ext::oneapi::experimental::sample_image<float>(
      texTarget, sycl::float2(x - 2.0f * dx, y));
  t1 -= sycl::ext::oneapi::experimental::sample_image<float>(
            texTarget, sycl::float2(x - 1.0f * dx, y)) *
        8.0f;
  t1 += sycl::ext::oneapi::experimental::sample_image<float>(
            texTarget, sycl::float2(x + 1.0f * dx, y)) *
        8.0f;
  t1 -= sycl::ext::oneapi::experimental::sample_image<float>(
      texTarget, sycl::float2(x + 2.0f * dx, y));
  t1 /= 12.0f;

  Ix[pos] = (t0 + t1) * 0.5f;
  //sycl::ext::oneapi::experimental::printf("Ix[%d]:%f\n",pos,Ix[pos]);

  // t derivative
  Iz[pos] = sycl::ext::oneapi::experimental::sample_image<float>(
                texTarget, sycl::float2(x, y)) -
            sycl::ext::oneapi::experimental::sample_image<float>(
                texSource, sycl::float2(x, y));

  //sycl::ext::oneapi::experimental::printf("Iz[%d]:%f\n",pos,Iz[pos]);

  // y derivative
  t0 = sycl::ext::oneapi::experimental::sample_image<float>(
      texSource, sycl::float2(x, y - 2.0f * dy));
  t0 -= sycl::ext::oneapi::experimental::sample_image<float>(
            texSource, sycl::float2(x, y - 1.0f * dy)) *
        8.0f;
  t0 += sycl::ext::oneapi::experimental::sample_image<float>(
            texSource, sycl::float2(x, y + 1.0f * dy)) *
        8.0f;
  t0 -= sycl::ext::oneapi::experimental::sample_image<float>(
      texSource, sycl::float2(x, y + 2.0f * dy));
  t0 /= 12.0f;

  t1 = sycl::ext::oneapi::experimental::sample_image<float>(
      texTarget, sycl::float2(x, y - 2.0f * dy));
  t1 -= sycl::ext::oneapi::experimental::sample_image<float>(
            texTarget, sycl::float2(x, y - 1.0f * dy)) *
        8.0f;
  t1 += sycl::ext::oneapi::experimental::sample_image<float>(
            texTarget, sycl::float2(x, y + 1.0f * dy)) *
        8.0f;
  t1 -= sycl::ext::oneapi::experimental::sample_image<float>(
      texTarget, sycl::float2(x, y + 2.0f * dy));
  t1 /= 12.0f;

  Iy[pos] = (t0 + t1) * 0.5f;

  //sycl::ext::oneapi::experimental::printf("Iy[%d]:%f\n",pos,Iy[pos]);

}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   image width
/// \param[in]  h   image height
/// \param[in]  s   image stride
/// \param[out] Ix  x derivative
/// \param[out] Iy  y derivative
/// \param[out] Iz  temporal derivative
///////////////////////////////////////////////////////////////////////////////
static void ComputeDerivatives(const float *I0, const float *I1, int w, int h,
                               int s, float *Ix, float *Iy, float *Iz) {
  sycl::range<3> threads(1, 6, 32);
  sycl::range<3> blocks(1, iDivUp(h, threads[1]), iDivUp(w, threads[2]));

  sycl::ext::oneapi::experimental::sampled_image_handle texSource, texTarget;
  dpct::image_data texRes;
  memset(&texRes, 0, sizeof(dpct::image_data));

  texRes.set_data_type(dpct::image_data_type::pitch);
  texRes.set_data_ptr((void *)I0);
  texRes.set_channel(dpct::image_channel::create<float>());
  texRes.set_x(w);
  texRes.set_y(h);
  texRes.set_pitch(s * sizeof(float));

  dpct::sampling_info texDescr;
  memset(&texDescr, 0, sizeof(dpct::sampling_info));

  texDescr.set(sycl::addressing_mode::mirrored_repeat,
               sycl::filtering_mode::linear,
               sycl::coordinate_normalization_mode::normalized);
  /*
  DPCT1062:7: SYCL Image doesn't support normalized read mode.
  */

  checkCudaErrors(DPCT_CHECK_ERROR(
      texSource = dpct::experimental::create_bindless_image(texRes, texDescr)));
  memset(&texRes, 0, sizeof(dpct::image_data));
  texRes.set_data_type(dpct::image_data_type::pitch);
  texRes.set_data_ptr((void *)I1);
  texRes.set_channel(dpct::image_channel::create<float>());
  texRes.set_x(w);
  texRes.set_y(h);
  texRes.set_pitch(s * sizeof(float));
  checkCudaErrors(DPCT_CHECK_ERROR(
      texTarget = dpct::experimental::create_bindless_image(texRes, texDescr)));

  /*
  DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
      sycl::nd_range<3>(blocks * threads, threads),
      [=](sycl::nd_item<3> item_ct1) {
        ComputeDerivativesKernel(w, h, s, Ix, Iy, Iz, texSource, texTarget,
                                 item_ct1);
      });
}
