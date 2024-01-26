//       $Id: bitmap.h 47053 2023-04-17 10:20:06Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/pfc/inc/pfc/bitmap.h $
// $Revision: 47053 $
//     $Date: 2023-04-17 12:20:06 +0200 (Mo., 17 Apr 2023) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: November 14, 2020
// Copyright: (c) 2021 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#pragma once

#undef  PFC_BITMAP_VERSION
#define PFC_BITMAP_VERSION "3.5.0"

#include <cstdint>
#include <cstring>

namespace pfc {

using byte_t  = std::uint8_t;
using dword_t = std::uint32_t;
using long_t  = std::int32_t;
using word_t  = std::uint16_t;

namespace bmp { namespace details {

#pragma pack (push, 1)
   struct BGR_3_t final {
      byte_t blue;
      byte_t green;
      byte_t red;
   };

   struct BGR_4_t final {
      union {
         BGR_3_t bgr_3;

         struct {
            byte_t blue;
            byte_t green;
            byte_t red;
         };
      };

      byte_t unused;
   };
#pragma pack (pop)

}   // namespace details

using pixel_t      = details::BGR_4_t;
using pixel_file_t = details::BGR_3_t;

} }   // namespace pfc::bmp

#if !defined __CUDACC__

#include <algorithm>
#include <cassert>
#include <concepts>
#include <format>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace pfc { namespace bmp {

using pixel_span_t      = std::span <pixel_t>;
using pixel_file_span_t = std::span <pixel_file_t>;

struct exception final : std::runtime_error {
   explicit exception (std::string const & m) : std::runtime_error {m} {
   }
};

}   // namespace bmp

class bitmap final {
   #pragma pack (push, 1)
      struct file_header_t final {
         word_t  type;            // file type; must be 0x4d42 (i.e. 'BM')
         dword_t size;            // size, in bytes, of the bitmap file
         word_t  reserved_1;      // reserved; must be 0
         word_t  reserved_2;      // reserved; must be 0
         dword_t offset;          // offset, in bytes, from the beginning of the 'file_header_t' to the bitmap bits
      };

      struct info_header_t final {
         dword_t size;            // number of bytes required by the structure
         long_t  width;           // width of the bitmap, in pixels
         long_t  height;          // height of the bitmap, in pixels
         word_t  planes;          // number of planes for the target device; must be 1
         word_t  bit_count;       // number of bits per pixel
         dword_t compression;     // type of compression; 0 for uncompressed RGB
         dword_t size_image;      // size, in bytes, of the span
         long_t  x_pels_pm;       // horizontal resolution, in pixels per meter
         long_t  y_pels_pm;       // vertical resolution, in pixels per meter
         dword_t clr_used;        // number of color indices in the color table
         dword_t clr_important;   // number of color indices that are considered important
      };
   #pragma pack (pop)

   static constexpr std::size_t size_file_header {sizeof (file_header_t)};
   static constexpr std::size_t size_info_header {sizeof (info_header_t)};
   static constexpr std::size_t size_pixel       {sizeof (bmp::pixel_t)};
   static constexpr std::size_t size_pixel_file  {sizeof (bmp::pixel_file_t)};

   static_assert (size_file_header == 14);
   static_assert (size_info_header == 40);

   public:
      static char const * version () {
         return PFC_BITMAP_VERSION;
      }

      bitmap () {
         create (0, 0);
      }

      explicit bitmap (std::size_t const width, std::size_t const height, bmp::pixel_span_t span = {}, bool const clear = true) {
         create (width, height, std::move (span), clear);
      }

      explicit bitmap (std::string const & filename) {
         from_file (filename);
      }

      bitmap (bitmap const & src) {
         create (src.m_width, src.m_height); memcpy (m_pixel_span, src.m_pixel_span); m_filename = src.m_filename;
      }

      bitmap (bitmap && tmp) noexcept
         : m_filename    {std::move (tmp.m_filename)}
         , m_file_header {tmp.m_file_header}
         , m_info_header {tmp.m_info_header}
         , m_height      {tmp.m_height}
         , m_width       {tmp.m_width}
         , m_p_image     {std::move (tmp.m_p_image)}
         , m_pixel_span  {std::move (tmp.m_pixel_span)} {
      }

     ~bitmap () = default;

      bitmap & operator = (bitmap const & rhs) {
         if (&rhs != this) {
            create (rhs.m_width, rhs.m_height); memcpy (m_pixel_span, rhs.m_pixel_span); m_filename = rhs.m_filename;
         }

         return *this;
      }

      bitmap & operator = (bitmap && tmp) noexcept {
         if (&tmp != this) {
            m_file_header = tmp.m_file_header;
            m_info_header = tmp.m_info_header;
            m_height      = tmp.m_height;
            m_width       = tmp.m_width;

            using std::swap;

            swap (m_filename,   tmp.m_filename);
            swap (m_p_image,    tmp.m_p_image);
            swap (m_pixel_span, tmp.m_pixel_span);
         }

         return *this;
      }

      auto begin () const {
         return std::begin (m_pixel_span);
      }

      auto end () const {
         return std::end (m_pixel_span);
      }

      auto * data () {
         return std::data (m_pixel_span);
      }

      auto const * data () const {
         return std::data (m_pixel_span);
      }

      auto & span () {
         return m_pixel_span;
      }

      auto const & span () const {
         return m_pixel_span;
      }

      auto size () const {
         return m_width * m_height;
      }

      auto size_bytes () const {
         return size () * size_pixel;
      }

      double aspect_ratio () const {
         return m_height > 0 ? 1.0 * m_width / m_height : 1.0;
      }

      auto const & filename () const {
         return m_filename;
      }

      auto const & width () const {
         return m_width;
      }

      auto const & height () const {
         return m_height;
      }

      auto & at (std::size_t const x, std::size_t const y) {
         return m_pixel_span[y * m_width + x];
      }

      auto const & at (std::size_t const x, std::size_t const y) const {
         return m_pixel_span[y * m_width + x];
      }

      void clear () {
         create (0, 0);
      }

      void create (std::size_t const width, std::size_t const height, bmp::pixel_span_t span = {}, bool const clear = true) {
         m_filename.clear ();

         m_width  = align_width (width);
         m_height = height;

         memset (m_file_header, 0);
         memset (m_info_header, 0);

         m_file_header.type   = 0x4d42;
         m_file_header.size   = size_file_header + size_info_header + static_cast <dword_t> (size ()) * size_pixel_file;
         m_file_header.offset = size_file_header + size_info_header;

         m_info_header.size       = size_info_header;
         m_info_header.width      = static_cast <long_t> (m_width);
         m_info_header.height     = static_cast <long_t> (m_height);
         m_info_header.planes     =  1;
         m_info_header.bit_count  = 24;
         m_info_header.size_image = static_cast <dword_t> (size ()) * size_pixel_file;

         if (!is_valid (m_file_header) || !is_valid (m_info_header))
            throw_exception ("Invalid pfc::bitmap header(s)");

         if (span.empty ()) {
            m_p_image    = std::make_unique <bmp::pixel_t []> (size ());
            m_pixel_span = {m_p_image.get (), size ()};
         } else {
            m_p_image    = nullptr;
            m_pixel_span = std::move (span);
         }

         if (std::size (m_pixel_span) < size ())
            throw_exception (std::format ("Pixel span too small (need space for {} pels, got {})", size (), std::size (m_pixel_span)));

         if (clear)
            memset (m_pixel_span, 0xff);
      }

      void from_file (std::string const & filename) {
         if (std::ifstream in {filename, std::ios_base::binary}) {
            turn_on_exceptions (in); clear ();

            file_header_t file_header {}; read (in, file_header);
            info_header_t info_header {}; read (in, info_header);

            if (!is_valid (file_header) || !is_valid (info_header))
               throw_exception (std::format ("Error reading bitmap headers from file '{}'", filename));

            create (info_header.width, info_header.height);

            if (span_from_stream (in, m_pixel_span, m_width) != size ())
               throw_exception (std::format ("Error reading bitmap data from file '{}'", filename));

            m_filename = filename;
         } else
            throw_exception (std::format ("Error opening bitmap file '{}' for reading", filename));
      }

      void to_file (std::string const & filename) const {
         if (std::ofstream out {filename, std::ios_base::binary}) {
            turn_on_exceptions (out);

            write (out, m_file_header);
            write (out, m_info_header);

            if (span_to_stream (out, m_pixel_span, m_width) != size ())
               throw_exception (std::format ("Error writing bitmap data to file '{}'", filename));

            m_filename = filename;
         } else
            throw_exception (std::format ("Error opening bitmap file '{}' for writing", filename));
      }

   private:
      static constexpr auto align_width (std::integral auto width) {
         while (width * 3 / 4 * 4 / 3 != width)
            ++width;

         return width;
      }

      static constexpr bool is_valid (file_header_t const & hdr) {
         return (hdr.offset     == size_file_header + size_info_header) &&
                (hdr.reserved_1 == 0) &&
                (hdr.reserved_2 == 0) &&
               !(hdr.size       <  size_file_header + size_info_header) &&
                (hdr.type       == 0x4d42);
      }

      static bool is_valid (info_header_t const & hdr) {
         return (hdr.bit_count     == 24) &&
                (hdr.clr_important ==  0) &&
                (hdr.clr_used      ==  0) &&
                (hdr.compression   ==  0) &&
                (hdr.planes        ==  1) &&
                (hdr.size          == size_info_header) &&
                (hdr.size_image    == static_cast <dword_t> (hdr.width * hdr.height * size_pixel_file)) &&
                (hdr.width         == align_width (hdr.width));
      }

      template <typename T> static void memcpy (std::span <T> & dst, std::span <T> const & src) {
         std::memcpy (std::data (dst), std::data (src), std::min (dst.size_bytes (), src.size_bytes ()));
      }

      template <typename T> static void memset (T & obj, byte_t const byte) {
         std::memset (&obj, byte, sizeof (T));
      }

      template <typename T> static void memset (std::span <T> & span, byte_t const byte) {
         std::memset (std::data (span), byte, span.size_bytes ());
      }

      template <typename T> static std::istream & read (std::istream & in, T & obj) {
         using char_t = std::remove_cvref_t <decltype (in)>::char_type; static_assert (sizeof (char_t) == 1); return in.read (reinterpret_cast <char_t *> (&obj), sizeof (T));
      }

      template <typename T> static std::istream & read (std::istream & in, std::span <T> & span) {
         using char_t = std::remove_cvref_t <decltype (in)>::char_type; static_assert (sizeof (char_t) == 1); return in.read (reinterpret_cast <char_t *> (std::data (span)), span.size_bytes ());
      }

      static std::size_t span_from_stream (std::istream & in, bmp::pixel_span_t const & span, std::size_t const chunk_size) {
         auto        buffer      {std::make_unique <bmp::pixel_file_t []> (chunk_size)};
         auto        buffer_span {bmp::pixel_file_span_t {buffer.get (), chunk_size}};
         std::size_t i           {0};
         std::size_t processed   {0};

         for (auto & pixel : span) {
            i %= chunk_size;

            if ((i == 0) && read (in, buffer_span))
               processed += chunk_size;

            pixel.bgr_3 = buffer_span[i++];
         }

         return processed;
      }

      static std::size_t span_to_stream (std::ostream & out, bmp::pixel_span_t const & span, std::size_t const chunk_size) {
         auto        buffer      {std::make_unique <bmp::pixel_file_t []> (chunk_size)};
         auto        buffer_span {bmp::pixel_file_span_t {buffer.get (), chunk_size}};
         std::size_t i           {0};
         std::size_t processed   {0};

         for (auto const & pixel : span) {
            buffer_span[i++] = pixel.bgr_3;

            if ((i == chunk_size) && write (out, buffer_span))
               processed += chunk_size;

            i %= chunk_size;
         }

         return processed;
      }

      static void throw_exception (std::string const & text) {
         throw bmp::exception {std::format ("PFC Bitmap v{}: {}.", PFC_BITMAP_VERSION, text)};
      }

      template <typename S> static S & turn_on_exceptions (S && io) {
         io.exceptions (std::ios::badbit /*| std::ios::eofbit*/ | std::ios::failbit); return io;
      }

      template <typename T> static std::ostream & write (std::ostream & out, T const & obj) {
         using char_t = std::remove_cvref_t <decltype (out)>::char_type; static_assert (sizeof (char_t) == 1); return out.write (reinterpret_cast <char_t *> (const_cast <T *> (&obj)), sizeof (T));
      }

      template <typename T> static std::ostream & write (std::ostream & out, std::span <T> const & span) {
         using char_t = std::remove_cvref_t <decltype (out)>::char_type; static_assert (sizeof (char_t) == 1); return out.write (reinterpret_cast <char_t *> (std::data (span)), span.size_bytes ());
      }

      std::string mutable               m_filename    {};
      file_header_t                     m_file_header {};
      info_header_t                     m_info_header {};
      std::size_t                       m_height      {};
      std::size_t                       m_width       {};
      std::unique_ptr <bmp::pixel_t []> m_p_image     {};
      bmp::pixel_span_t                 m_pixel_span  {};
};

}   // namespace pfc

#endif   // __CUDACC__
