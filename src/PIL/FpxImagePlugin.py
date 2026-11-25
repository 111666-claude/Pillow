#
# THIS IS WORK IN PROGRESS
#
# The Python Imaging Library.
# $Id$
#
# FlashPix support for PIL
#
# History:
# 97-01-25 fl   Created (reads uncompressed RGB images only)
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1997.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

__lazy_modules__ = {"PIL._binary"}

import olefile

from . import Image, ImageFile
from ._binary import i32le as i32

# we map from colour field tuples to (mode, rawmode) descriptors
MODES = {
    # opacity
    (0x00007FFE,): ("A", "L"),
    # monochrome
    (0x00010000,): ("L", "L"),
    (0x00018000, 0x00017FFE): ("RGBA", "LA"),
    # photo YCC
    (0x00020000, 0x00020001, 0x00020002): ("RGB", "YCC;P"),
    (0x00028000, 0x00028001, 0x00028002, 0x00027FFE): ("RGBA", "YCCA;P"),
    # standard RGB (NIFRGB)
    (0x00030000, 0x00030001, 0x00030002): ("RGB", "RGB"),
    (0x00038000, 0x00038001, 0x00038002, 0x00037FFE): ("RGBA", "RGBA"),
}


#
# --------------------------------------------------------------------


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(olefile.MAGIC)


##
# Image plugin for the FlashPix images.


class FpxImageFile(ImageFile.ImageFile):
    format = "FPX"
    format_description = "FlashPix"

    def _open(self) -> None:
        #
        # read the OLE directory and see if this is a likely
        # to be a FlashPix file

        assert self.fp is not None
        try:
            self.ole = olefile.OleFileIO(self.fp)
        except OSError as e:
            msg = "not an FPX file; invalid OLE file"
            raise SyntaxError(msg) from e

        root = self.ole.root
        if not root or root.clsid != "56616700-C154-11CE-8553-00AA00A1F95B":
            msg = "not an FPX file; bad root CLSID"
            raise SyntaxError(msg)

        self._open_index(1)

    def _validate_property_set(self, path: list[str]) -> None:
        # Parse just enough of the OLE property set to make sure no vector
        # property claims more elements than the stream could contain.
        # olefile's getproperties() trusts the count and loops over every
        # element, so an inflated value can hang the decoder.
        data = self.ole.openstream(path).read()

        if len(data) < 48:
            msg = "truncated Image Contents property set"
            raise OSError(msg)

        section_count = i32(data, 24)
        if section_count <= 0 or 28 + section_count * 20 > len(data):
            msg = "invalid property set section count"
            raise OSError(msg)

        for section_index in range(section_count):
            locator = 28 + section_index * 20
            section_offset = i32(data, locator + 16)
            if section_offset + 8 > len(data):
                msg = "invalid property section offset"
                raise OSError(msg)

            section_size = i32(data, section_offset)
            section_end = section_offset + section_size
            if section_size <= 8 or section_end > len(data):
                msg = "invalid property section size"
                raise OSError(msg)

            property_count = i32(data, section_offset + 4)
            directory_end = section_offset + 8 + property_count * 8
            if property_count <= 0 or directory_end > section_end:
                msg = "invalid property count"
                raise OSError(msg)

            for property_index in range(property_count):
                entry = section_offset + 8 + property_index * 8
                value_offset = i32(data, entry + 4)
                if value_offset >= section_size - 4:
                    msg = "invalid property value offset"
                    raise OSError(msg)

                property_type = i32(data, section_offset + value_offset)
                if property_type & 0x1000:  # VT_VECTOR
                    # The element count is the first 32-bit value; every
                    # element needs at least one byte within the section.
                    count_offset = section_offset + value_offset + 4
                    if count_offset + 4 > section_end:
                        msg = "truncated vector property"
                        raise OSError(msg)
                    element_count = i32(data, count_offset)
                    if element_count > section_end - count_offset - 4:
                        msg = "vector property extends past the property set"
                        raise OSError(msg)

    def _open_index(self, index: int = 1) -> None:
        #
        # get the Image Contents Property Set

        property_path = [f"Data Object Store {index:06d}", "\005Image Contents"]

        # Validate the property set before handing it to olefile: a malformed
        # vector count can make olefile's parser loop for a very long time.
        self._validate_property_set(property_path)

        prop = self.ole.getproperties(property_path)

        # size (highest resolution)

        width = prop.get(0x1000002)
        height = prop.get(0x1000003)
        if not isinstance(width, int) or not isinstance(height, int):
            msg = "invalid image size"
            raise OSError(msg)
        # Values are read as unsigned 32-bit integers, so a negative field
        # shows up as a very large number; reject anything outside the range
        # of a sensible positive dimension.
        if not 0 < width <= 0x7FFFFFFF or not 0 < height <= 0x7FFFFFFF:
            msg = "image dimensions must be positive"
            raise OSError(msg)

        # Reject absurdly large declared dimensions before they can drive
        # allocations or tile-grid calculations.
        Image._decompression_bomb_check((width, height), error_only=True)
        self._size = width, height
        size = max(self.size)
        i = 1
        while size > 64:
            size = size // 2
            i += 1
        self.maxid = i - 1

        # mode.  instead of using a single field for this, flashpix
        # requires you to specify the mode for each channel in each
        # resolution subimage, and leaves it to the decoder to make
        # sure that they all match.  for now, we'll cheat and assume
        # that this is always the case.

        id = self.maxid << 16

        s = prop.get(0x2000002 | id)

        if not isinstance(s, bytes) or len(s) < 8:
            msg = "invalid colour space descriptor"
            raise OSError(msg)

        bands = i32(s, 4)
        if bands <= 0 or bands > 4 or len(s) < 8 + bands * 4:
            msg = "Invalid number of bands"
            raise OSError(msg)

        # note: for now, we ignore the "uncalibrated" flag
        colors = tuple(i32(s, 8 + i * 4) & 0x7FFFFFFF for i in range(bands))

        try:
            self._mode, self.rawmode = MODES[colors]
        except KeyError as e:
            msg = "unsupported colour space"
            raise OSError(msg) from e

        # load JPEG tables, if any
        self.jpeg = {}
        for i in range(256):
            id = 0x3000001 | (i << 16)
            if id in prop:
                self.jpeg[i] = prop[id]

        self._open_subimage(1, self.maxid)

    def _open_subimage(self, index: int = 1, subimage: int = 0) -> None:
        #
        # setup tile descriptors for a given subimage

        stream = [
            f"Data Object Store {index:06d}",
            f"Resolution {subimage:04d}",
            "Subimage 0000 Header",
        ]

        fp = self.ole.openstream(stream)
        stream_size = self.ole.get_size(stream)

        # skip prefix
        fp.read(28)

        # header stream
        s = fp.read(36)
        if len(s) < 36:
            msg = "subimage header is truncated"
            raise OSError(msg)

        size = i32(s, 4), i32(s, 8)
        tilecount = i32(s, 12)
        xtile, ytile = i32(s, 16), i32(s, 20)
        if xtile != 64 or ytile != 64:
            msg = "Tile must be 64 pixels by 64 pixels"
            raise ValueError(msg)

        # channels = i32(s, 24)
        offset = i32(s, 28)
        length = i32(s, 32)

        if size != self.size:
            msg = "subimage mismatch"
            raise OSError(msg)

        xsize, ysize = size

        # The tile directory is a fixed-size array of "tilecount" entries of
        # "length" bytes, located at "offset" within the header stream.
        # Validate every field before using it so that a malformed file can
        # neither drive an oversized read nor walk past the stream.
        if tilecount <= 0:
            msg = "invalid tile count"
            raise OSError(msg)
        if length < 16:
            # entries are indexed at offsets 0, 8, 12, 14 and 15
            msg = "invalid tile entry length"
            raise OSError(msg)
        if offset < 36:
            msg = "tile directory overlaps the subimage header"
            raise OSError(msg)

        directory_position = 28 + offset
        if directory_position > stream_size:
            msg = "tile directory starts past the subimage header"
            raise OSError(msg)

        directory_size = tilecount * length
        if directory_position + directory_size > stream_size:
            msg = "tile directory extends past the subimage header"
            raise OSError(msg)

        # The declared tile count must match the grid implied by the
        # subimage dimensions and the 64x64 tile size (edge tiles may cover
        # fewer pixels, but still occupy a grid cell).
        grid_x = (xsize + xtile - 1) // xtile
        grid_y = (ysize + ytile - 1) // ytile
        if tilecount != grid_x * grid_y:
            msg = "tile count does not match the image grid"
            raise OSError(msg)

        # get tile descriptors
        fp.seek(28 + offset)
        s = fp.read(directory_size)
        if len(s) < directory_size:
            msg = "tile directory is truncated"
            raise OSError(msg)

        x = y = 0
        self.tile = []

        for i in range(0, directory_size, length):
            x1 = min(xsize, x + xtile)
            y1 = min(ysize, y + ytile)

            compression = i32(s, i + 8)

            if compression == 0:
                self.tile.append(
                    ImageFile._Tile(
                        "raw",
                        (x, y, x1, y1),
                        i32(s, i) + 28,
                        self.rawmode,
                    )
                )

            elif compression == 1:
                # FIXME: the fill decoder is not implemented
                self.tile.append(
                    ImageFile._Tile(
                        "fill",
                        (x, y, x1, y1),
                        i32(s, i) + 28,
                        (self.rawmode, s[12:16]),
                    )
                )

            elif compression == 2:
                internal_color_conversion = s[14]
                jpeg_tables = s[15]
                rawmode = self.rawmode

                if internal_color_conversion:
                    # The image is stored as usual (usually YCbCr).
                    if rawmode == "RGBA":
                        # For "RGBA", data is stored as YCbCrA based on
                        # negative RGB. The following trick works around
                        # this problem :
                        jpegmode, rawmode = "YCbCrK", "CMYK"
                    else:
                        jpegmode = None  # let the decoder decide

                else:
                    # The image is stored as defined by rawmode
                    jpegmode = rawmode

                self.tile.append(
                    ImageFile._Tile(
                        "jpeg",
                        (x, y, x1, y1),
                        i32(s, i) + 28,
                        (rawmode, jpegmode),
                    )
                )

                # FIXME: jpeg tables are tile dependent; the prefix
                # data must be placed in the tile descriptor itself!

                if jpeg_tables:
                    try:
                        self.tile_prefix = self.jpeg[jpeg_tables]
                    except KeyError as e:
                        msg = "missing JPEG tables"
                        raise OSError(msg) from e

            else:
                msg = "unknown/invalid compression"
                raise OSError(msg)

            x += xtile
            if x >= xsize:
                x, y = 0, y + ytile
                if y >= ysize:
                    break  # isn't really required

        assert self.fp is not None
        self.stream = stream
        self._fp = self.fp
        self.fp = None

    def load(self) -> Image.core.PixelAccess | None:
        if not self.fp:
            self.fp = self.ole.openstream(self.stream[:2] + ["Subimage 0000 Data"])

        return ImageFile.ImageFile.load(self)

    def close(self) -> None:
        self.ole.close()
        super().close()

    def __exit__(self, *args: object) -> None:
        self.ole.close()
        super().__exit__()


#
# --------------------------------------------------------------------


Image.register_open(FpxImageFile.format, FpxImageFile, _accept)

Image.register_extension(FpxImageFile.format, ".fpx")
