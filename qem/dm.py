"""
Reading/writting dm3/dm4 files
Pavel Potapov, temDM, 2022

can be used under
GNU Lesser General Public License  
"""

import struct

import numpy as np


def dm_load(filename, index=0, prt=False) -> tuple:
    # index shows which image is needed,prt: print intermediate output
    with open(filename, "rb") as f:
        dm = read_DM(f, prt)  # initialize red class
        dm.read_file()  # read as dictionary

        dm.make_image_list()  # extract images, remove thumbnail
        image, dimensions, calibration = dm.get_data_array(
            index
        )  # output image[index],calibration
        metadata = dm.get_matadata(index)  # output metadata

        f.close()

        return image, dimensions, calibration, metadata


def dm_load_as_tags(filename, index=0) -> dict:  # for testing purpose

    with open(filename, "rb") as f:
        dm = read_DM(f, prt=True)
        dm.read_file()

        f.close()

        return dm.tags_dict


def dm_store(
    filename,
    Image,
    dimensions: dict,
    dm_ext: str,
    calibration=None,
    meta=None,
    prt=False,
):
    # numpy type Image expected, dm_ext: 'dm3' or 'dm4'
    # store or not calibration / metadata, prt: print intermediate output
    im_type = str(Image.dtype)
    im_size = Image.size
    dim = Image.ndim

    dm = write_DM(im_type, im_size, dm_ext, prt)  # initialize write class
    dm.must_tags(
        dim, dimensions, calibration, meta
    )  # obligatory tags incl. optional calibrations and meta
    dm.write_to_buffer()  # convert to buffer

    with open(filename, "wb") as f:
        f.write(dm.b_header)  # write file header
        f.write(dm.b_before)  # write part before image
        f.write(Image.tobytes())  # write image
        f.write(dm.b_after)  # write part after image
        f.write(set_uint64(0, "little"))  # write 8 zero bytes (end marker)

        f.close()


def dm_store_must_tags(
    Image, dimensions: dict, dm_ext: str, calibration=None, meta=None
) -> dict:  # for testing purpose
    # numpy type Image expected
    im_type = str(Image.dtype)
    im_size = Image.size

    prt = True
    dm = write_DM(im_type, im_size, dm_ext, prt)

    must_tags = dm.must_tags(
        dimensions, calibration, meta
    )  # obligatory tags incl. optional calibrations and meta

    return must_tags


"""
ELEMENTAL READING BLOCKS
"""


def set_endian(endian: str) -> str:
    allowed = {"little", "big", "L", "B", "l", "b"}
    if endian not in allowed:
        raise ValueError("wrong endian")

    if endian == "big" or endian == "B" or endian == "b":
        return ">"

    if endian == "little" or endian == "L" or endian == "l":
        return "<"


def get_Specified_value(f, length: int, endian: str, Identifier: str):
    buffer = f.read(length)
    end = set_endian(endian)
    return struct.unpack(end + Identifier, buffer)[0]  # always tuple even if one value


def get_int8(f, endian: str) -> int:  # get 1-byte integer from buffer f
    identifier = "b"
    length = 1
    return get_Specified_value(f, length, endian, identifier)


def get_uint8(f, endian: str) -> int:  # get 1-byte unsigned integer from buffer f
    identifier = "B"
    length = 1
    return get_Specified_value(f, length, endian, identifier)


def get_int16(f, endian: str) -> int:  # get 2-byte integer from buffer f
    identifier = "h"
    length = 2
    return get_Specified_value(f, length, endian, identifier)


def get_uint16(f, endian: str) -> int:  # get 2-byte unsigned integer from buffer f
    identifier = "H"
    length = 2
    return get_Specified_value(f, length, endian, identifier)


def get_int32(f, endian: str) -> int:  # get 4-byte integer from buffer f
    identifier = "l"
    length = 4
    return get_Specified_value(f, length, endian, identifier)


def get_uint32(f, endian: str) -> int:  # get 4-byte unsigned integer from buffer f
    identifier = "L"
    length = 4
    return get_Specified_value(f, length, endian, identifier)


def get_int64(f, endian: str) -> int:  # get 8-byte integer from buffer f
    identifier = "q"
    length = 8
    return get_Specified_value(f, length, endian, identifier)


def get_uint64(f, endian: str) -> int:  # get 8-byte unsigned integer from buffer f
    identifier = "Q"
    length = 8
    return get_Specified_value(f, length, endian, identifier)


def get_float32(f, endian: str) -> float:  # get 4-byte float from buffer f
    identifier = "f"
    length = 4
    return get_Specified_value(f, length, endian, identifier)


def get_float64(f, endian: str) -> float:  # get 8-byte float from buffer f
    identifier = "d"
    length = 8
    return get_Specified_value(f, length, endian, identifier)


def get_char(f, endian: str) -> str:  # get 1-byte char from buffer f
    identifier = "c"
    length = 1
    return get_Specified_value(f, length, endian, identifier)


"""
ELEMENTAL WRITTING BLOCKS
"""


def set_int8(value: int, endian: str):  # set 1-byte integer
    identifier = "b"
    return struct.pack(set_endian(endian) + identifier, value)


def set_uint8(value: int, endian: str):  # set 1-byte unsigned integer
    identifier = "B"
    return struct.pack(set_endian(endian) + identifier, value)


def set_int16(value: int, endian: str):  # set 2-byte  integer
    identifier = "h"
    return struct.pack(set_endian(endian) + identifier, value)


def set_uint16(value: int, endian: str):  # set 2-byte unsigned integer
    identifier = "H"
    return struct.pack(set_endian(endian) + identifier, value)


def set_int32(value: int, endian: str):  # set 4-byte integer
    identifier = "l"
    return struct.pack(set_endian(endian) + identifier, value)


def set_uint32(value: int, endian: str):  # set 4-byte unsigned integer
    identifier = "L"
    return struct.pack(set_endian(endian) + identifier, value)


def set_int64(value: int, endian: str):  # set 8-byte integer
    identifier = "q"
    return struct.pack(set_endian(endian) + identifier, value)


def set_uint64(value: int, endian: str):  # set 8-byte unsigned integer
    identifier = "Q"
    return struct.pack(set_endian(endian) + identifier, value)


def set_float32(value: float, endian: str):  # set 4-byte float
    identifier = "f"
    return struct.pack(set_endian(endian) + identifier, value)


def set_float64(value: float, endian: str):  # set 8-byte float
    identifier = "d"
    return struct.pack(set_endian(endian) + identifier, value)


def set_char(valueStr: float, endian: str):  # set 1-byte char
    identifier = "c"
    value = bytes(valueStr, "utf-8")
    return struct.pack(set_endian(endian) + identifier, value)


"""
Dictionary to convert types
"""

type_dict_TC = {
    # type_code as a keyword
    # type_code: (bytes, NumPy type, DM stream code, reader, writer,NumPy setter, DM image type)
    "b": (1, "int8", 10, get_int8, set_int8, np.int8, 9),
    "B": (1, "uint8", 8, get_uint8, set_uint8, np.uint8, 6),
    "i2": (2, "int16", 2, get_int16, set_int16, np.int16, 1),
    "h": (2, "int16", 2, get_int16, set_int16, np.int16, 1),
    "u2": (2, "uint16", 4, get_uint16, set_uint16, np.uint16, 10),
    "H": (2, "uint16", 4, get_uint16, set_uint16, np.uint16, 10),
    "i4": (4, "int32", 3, get_int32, set_int32, np.int32, 7),
    "i": (4, "int32", 3, get_int32, set_int32, np.int32, 7),
    "l": (4, "int32", 3, get_int32, set_int32, np.int32, 7),
    "u4": (4, "uint32", 5, get_uint32, set_uint32, np.uint32, 11),
    "L": (4, "uint32", 5, get_uint32, set_uint32, np.uint32, 11),
    "q": (8, "int64", 11, get_int64, set_int64, np.int64, 0),
    "Q": (8, "uint64", 12, get_uint64, set_uint64, np.uint64, 0),
    "f4": (4, "float32", 6, get_float32, set_float32, np.float32, 2),
    "f": (4, "float32", 6, get_float32, set_float32, np.float32, 2),
    "f8": (8, "float64", 7, get_float64, set_float64, np.float64, 12),
    "d": (8, "float64", 7, get_float64, set_float64, np.float64, 12),
    "c": (8, "complex64", 0, None, None, np.complex64, 3),
    "c8": (8, "complex64", 0, None, None, np.complex64, 3),
    "c16": (16, "complex128", 0, None, None, np.complex128, 13),
}


def NP_number_types() -> tuple:  # supported numPy types of numbers
    return (
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )


def tag_counts() -> tuple:
    counts = ()
    for i in range(10):
        counts += (str(i),)
    return counts


"""
CLASS read dm file
"""


class read_DM(object):

    def __init__(self, f, prt):
        self.f = f  # file stream
        self.dm_ext = None  # 3 or 4
        self.endian = None  # usually little
        self.tags_dict = {}  # main dictionary to store all dm info
        self.image_list = list()  # dictionary where the image is
        self.meta_list = {}  # dictionary where metadata is
        self.n_images = 0  # number of images stored in file
        self.init_dict()  # initialize types table
        self.print = prt  # print out or not intermediate outputs

    def init_dict(self):
        """
        #reverse dictionary such as DM stream code is a key
        """
        self.type_dict_DM = dict(
            (v[2], (v[0], v[1], k, v[3], v[4], v[5])) for k, v in type_dict_TC.items()
        )
        # add extra DM codes
        self.type_dict_DM[9] = (
            1,
            "char",
            "b",
            get_char,
            set_char,
            None,
            0,
        )  # 1byte character used for tag labels
        self.type_dict_DM[15] = (
            None,
            "structure",
            "struct",
            self.read_structure,
            None,
            None,
            0,
        )
        self.type_dict_DM[18] = (None, "string", "c", self.read_string, None, None, 0)
        self.type_dict_DM[20] = (None, "array", "array", self.read_array, None, None, 0)

    def read_file(self):
        self.f.seek(0)
        self.file_header()
        self.parse_tags_to_dict(group_name="", group_dict=self.tags_dict, root=True)

    def file_header(self):
        """
        Read dm format (extension) and endian
        1:dm ext (dm3 or dm4)
        2:length in bytes
        3: endian, 0-big, 1-little (only for INSIDE tags)
        """
        self.dm_ext = get_int32(self.f, "big")

        length = self.read_DMvalue()  # total file length (not used)

        byte_ordering = get_int32(self.f, "big")
        if self.print:
            print("dm", self.dm_ext, " length ", length, " ordering ", byte_ordering)

        if bool(byte_ordering):
            self.endian = "little"
        else:
            self.endian = "big"

    def read_DMvalue(self) -> int:
        """
        diferent formats for dm3 and dm4
        """
        if self.dm_ext == 4:
            return get_int64(self.f, "big")
        else:
            return get_int32(self.f, "big")

    def group_header(self, group_name="", root=False) -> tuple:
        """
        Read header of TagGroup
        1:length of the whole group (in dm4 only)
        2:is group sorted (meaning unknown)
        3:is group open
        4:number of tags in group
        """
        l_group = 0
        if self.dm_ext == 4 and root is False:
            l_group = get_uint64(self.f, "big")
        is_sorted = get_int8(self.f, "big")
        is_open = get_int8(self.f, "big")
        n_tags = self.read_DMvalue()

        return bool(is_sorted), bool(is_open), n_tags, l_group

    def name_header(self) -> tuple:
        """
        Read name of tag and its code
        1:tag_code =21 if simple Tag, =20 if TagGroup branch
        2:length of tag name
        3:tag_name - names of Tag or TagGroup
        """
        tag_code = get_int8(self.f, "big")
        tag_name_length = get_int16(self.f, "big")
        tag_name = self.read_string(tag_name_length)

        if tag_code == 21:
            self.check_delimiter(3)  # check the tag marker

        return tag_code, tag_name

    def check_delimiter(self, deviation: int):
        # check if the tag marker is at right place after the name tag
        if self.dm_ext == 4:
            get_int64(self.f, "big")
        marker = self.read_string(4)  # read '%%%%' marker
        if marker != "%%%%":  # name length can be inaccurate!
            if self.print:
                print("marker:", marker)
            self.f.seek(-(4 + deviation), 1)  # set stream back

            markers = 0  # markers count
            # find th eright position
            for _i in range(4 + 2 * deviation):
                ch = self.read_string(1)  # read '%' marker
                if self.print:
                    print(ch)
                if ch == "%":
                    markers += 1
                    if markers == 4:
                        break

        # return to the right end of name tag
        self.f.seek(-4, 1)
        if self.dm_ext == 4:
            self.f.seek(-8, 1)

    def tag_header(self) -> tuple:
        """
        Read type of Tag and its length (in dm4 only) and length definition
        1:length of tag (in dm4 only)
        1:marker %%%% indicating the beginning of a tag
        2:definition of length
        """
        tag_length = 0
        if self.dm_ext == 4:  # tag lengh is stored in dm4 only
            tag_length = get_int64(self.f, "big")

        self.read_string(4)  # read '%%%%' marker
        length_definitions = self.read_DMvalue()

        return length_definitions, tag_length

    def parse_tags_to_dict(self, group_name: str, group_dict: dict, root=False):
        """
        recursive funcion to scroll through DM file tags structure
        """
        is_sorted, is_open, n_tags, l_group = self.group_header(
            group_name=group_name, root=root
        )

        if self.print:
            print(
                "group sorted/open:",
                is_sorted,
                is_open,
                "length",
                l_group,
                "offset",
                self.f.tell(),
                "number of tags ",
                n_tags,
            )
        if is_sorted is False:
            group_dict["sorted"] = False

        index = 0
        for tag in range(n_tags):
            tag_code, tag_name = self.name_header()
            if self.print:
                print("   ", tag, "Name:", tag_name, "      code:", tag_code)
            if root is False and not tag_name:  # if tag_name=='', they are numbered
                tag_name = str(index)
                index += 1

            if tag_code == 21:  # Tag
                length_definitions, tag_length = self.tag_header()
                dtype = self.read_DMvalue()  # common tag for all data types
                if self.print:
                    print(
                        "   tag_length",
                        tag_length,
                        "Length definition",
                        length_definitions,
                        "offset",
                        self.f.tell(),
                        "dtype",
                        dtype,
                    )
                if length_definitions == 1:  # Simple type
                    data = self.read_data(dtype)

                elif length_definitions == 2:  # String
                    if dtype != 18:
                        raise IOError(
                            "according length definition (2) it should be type 18, not",
                            dtype,
                        )
                    string_length = self.read_DMvalue()  # extra tag for strings
                    data = self.read_string(string_length)

                elif length_definitions == 3:  # Array
                    if dtype != 20:  # Should be 20 for array
                        raise IOError(
                            "according length definition (3) it should be type 20, not",
                            dtype,
                        )
                    eltype = self.read_DMvalue()  # extra tags for arrays
                    size = self.read_DMvalue()
                    if self.print:
                        print("                 eltypes", eltype, "size", size)
                    if (
                        group_name == "ImageData" and tag_name == "Data"
                    ):  # this is image array
                        offset = self.f.tell()
                        if self.print:
                            print("ImageDataArray", dtype, " ", eltype, " size ", size)
                        pixel_depth = self.type_dict_DM[eltype][0]

                        size_bytes = pixel_depth * size
                        self.f.seek(size_bytes, 1)  # skip array
                        data = (
                            offset,
                            size,
                            eltype,
                        )  # store its offset,size and type instead

                    else:  # this is some other array
                        data = self.read_array(size, eltype)

                elif length_definitions > 3:  # composed data
                    if dtype == 15:  #  structure
                        types = (
                            self.struct_types()
                        )  # types of all elements in structure
                        if self.print:
                            print("TupleType", types)
                        data = self.read_structure(types)
                    elif dtype == 20:  # array of something
                        eltype = self.read_DMvalue()  # type of array
                        if eltype == 15:  # array of structures
                            types = (
                                self.struct_types()
                            )  # types of all elements in structure
                            size = self.read_DMvalue()
                            data = self.read_array(
                                size, eltype, inserted={"types": types}
                            )
                        elif eltype == 18:  # array of strings
                            string_length = self.read_DMvalue()
                            size = self.read_DMvalue()
                            data = self.read_array(
                                size, eltype, inserted={"length": string_length}
                            )
                        elif eltype == 20:  # array of arrays
                            each_type = self.read_DMvalue()
                            each_size = self.read_DMvalue()
                            size = self.read_DMvalue()
                            data = self.read_array(
                                size,
                                each_type,  # each_type=eltype ??
                                inserted={"size": each_size},
                            )
                    else:
                        raise IOError(
                            "length definition",
                            length_definitions,
                            "or dtype",
                            dtype,
                            "are incorrect",
                        )

                else:
                    raise IOError("Length definition must be non-zero !!")

                group_dict[tag_name] = data

            elif tag_code == 20:  # TagGroup
                group_dict[tag_name] = {}
                self.parse_tags_to_dict(
                    group_name=tag_name, group_dict=group_dict[tag_name]
                )

            else:
                raise IOError(tag, "ERROR", tag_code)

            if group_name == "ImageList":
                self.n_images = index  # total number of images stored

        if self.print:
            print("End of Group", self.f.tell())

    def make_image_list(self):
        """
        remove thumnail item from dictionary
        """
        del self.tags_dict["ImageList"]["sorted"]  # this tag disturbs image numbering

        for index in range(self.n_images):
            Dtype = self.tags_dict["ImageList"][str(index)]["ImageData"]["DataType"]
            if Dtype == 23:  # this is thumbnail
                del self.tags_dict["ImageList"][str(index)]  # remove
                self.n_images -= 1  # total number of iages reduced
        """
        turn dictionary into intermediate image list  [(index,image_dictionaries)]
        """
        image_list0 = list(self.tags_dict["ImageList"].items())

        """
        restore image list numeration [image_dictionaries]
        """
        for index in range(self.n_images):
            self.image_list.append(image_list0[index][1])

    def get_data_array(self, index: int) -> tuple:
        """
        read array associated with an image
        """
        offset, size, eltype = self.image_list[index]["ImageData"]["Data"]

        dtype = self.type_dict_DM[eltype][1]
        self.f.seek(offset)
        image = np.fromfile(self.f, dtype=dtype, count=size)

        """
        read information
        """
        dim_dict = self.image_list[index]["ImageData"][
            "Dimensions"
        ]  # dictionary with width,height...
        if "sorted" in dim_dict:
            del dim_dict["sorted"]
        n_dim = len(dim_dict)

        cal_dict = None
        if "Calibrations" in self.image_list[index]["ImageData"]:
            cal_dict = self.image_list[index]["ImageData"]["Calibrations"]["Dimension"]
            cal_dict["Intensity"] = self.image_list[index]["ImageData"]["Calibrations"][
                "Brightness"
            ]
            #'Brightness' is renamed to 'Intensity' and kept similar to dimensional items
            if "sorted" in cal_dict:
                del cal_dict["sorted"]

        """
        reshape according dimensions
        """
        if n_dim == 1:
            image = image.reshape(dim_dict["0"])
        elif n_dim == 2:
            image = image.reshape(dim_dict["1"], dim_dict["0"])
        elif n_dim == 3:
            image = image.reshape(dim_dict["2"], dim_dict["1"], dim_dict["0"])

        return image, dim_dict, cal_dict

    def read_data(self, dtype: int):
        """
        elemental data piece
        """
        data = self.type_dict_DM[dtype][3](self.f, self.endian)
        data = self.type_dict_DM[dtype][5](data)  # force numpy, not Python data type

        return data

    def read_string(self, length: int) -> str:
        """
        1byte char (DM code 9) seems to be used for tag names only
        for the strings, array of 2byte symbols (DM code 4) mostly used
        """
        data = b""
        for _char in range(length):
            data += get_char(self.f, self.endian)

        try:
            data = data.decode("utf8")
        except BaseException:
            # often latin-1 encoding
            data = data.decode("latin-1", errors="ignore")

        return data

    def struct_types(self) -> tuple:
        """
        read the tuple with types of structure elements
        (type_0,type_1,...,type_n-1)
        """
        self.read_DMvalue()  # should be zero
        n_notes = self.read_DMvalue()
        types = ()
        for _type in range(n_notes):
            self.read_DMvalue()  # should be zero
            types += (self.read_DMvalue(),)

        return types

    def read_structure(self, types: tuple) -> tuple:
        """
        read a tuple with different (not necesarily same type) data pieces
        """
        field_value = []
        for dtype in types:
            data = self.type_dict_DM[dtype][3](self.f, self.endian)
            data = self.type_dict_DM[dtype][5](
                data
            )  # force numpy, not Python data type
            field_value.append(data)

        return tuple(field_value)

    def read_array(self, size: int, eltype: int, inserted=None):
        """
        read non-image array
        """
        reader = self.type_dict_DM[eltype][3]

        if eltype == 15 or eltype == 18 or eltype == 20:
            # array of structures, strings or arrays
            data = [reader(**inserted) for element in range(size)]
        else:  # array of numbers
            if eltype == 4:  # array of char, i.e. string (often in DM files)
                data = ""
                for _i in range(size):
                    data += chr(reader(self.f, self.endian))
            else:
                data = [reader(self.f, self.endian) for element in range(size)]
                data = self.type_dict_DM[eltype][5](
                    data
                )  # force numpy, not Python data type

        return data

    def get_matadata(self, index) -> dict:
        meta_dict = None
        if "ImageTags" in self.image_list[index]:
            meta_dict = self.image_list[index]["ImageTags"]

        return meta_dict


"""
CLASS store dm file
"""


class write_DM(object):

    def __init__(self, im_type, im_size, dm_ext, prt):
        if dm_ext == "dm4":
            self.dm_ext = 4
        else:
            self.dm_ext = 3
        self.endian = "little"
        self.init_dict()  # initialize types table
        self.type = self.type_dict_NP[im_type][2]  # DM type code
        self.Itype = self.type_dict_NP[im_type][6]  # DM image type
        self.depth = self.type_dict_NP[im_type][0]  # byte depth for image
        self.size = im_size  # number of elements in image
        self.tags_dict = {}  # main tags dictionary
        self.group_stack = [
            -1
        ]  # stack to keep information about group offset and group size
        # -1 signals about root group (no length headers)
        self.tag_address = 0  # address where the tag length should be stored
        self.print = prt  # print out the intermediate output

    def init_dict(self):
        # reverse dictionary such as NumPy type is a key
        self.type_dict_NP = dict(
            (v[1], (v[0], k, v[2], v[3], v[4], v[5], v[6]))
            for k, v in type_dict_TC.items()
        )

        self.type_dict_NP["str"] = (
            2,
            "H",
            4,
            get_uint16,
            set_uint16,
            np.uint16,
            0,
        )  # strings are stored as arrays of 2byte 'H' symbols

    def get_type(self, value) -> str:
        Ttype = type(value).__name__  # numPy types expected

        # force flexible Python types to hard numPy types
        if Ttype == "int":
            if value >= 0:
                Ttype = "uint32"
            else:
                Ttype = "int32"
        if Ttype == "float":
            Ttype = "float32"
        if Ttype == "bool":
            Ttype = "uint8"

        return Ttype

    def set_DMvalue(self, value: int) -> int:
        """
        diferent int length for dm3 and dm4
        """
        if self.dm_ext == 4:
            return set_int64(value, "big")
        else:
            return set_int32(value, "big")

    def must_tags(
        self, dim: int, dimensions: dict, calibration=None, metadata=None
    ) -> dict:
        """
        according Bernhard Schaffer these Tags are obligatory for any dm file
        """
        im_data_dict = {}
        if calibration is not None:
            calibrationC = calibration.copy()
            int_dict = calibrationC["Intensity"]
            del calibrationC["Intensity"]
            calibr_dict = {
                "Calibrations": {
                    "Brightness": int_dict,
                    "Dimension": calibrationC,  # dimension(s)!!!
                    "DisplayCalibratedUnits": True,
                }
            }
            calibr_dict["Calibrations"]["Dimension"]["sorted"] = False  # temdm tag
            im_data_dict.update(calibr_dict)

        im_data_dict["DataImageArray"] = None  # signal code for image data
        im_data_dict["DataType"] = self.Itype
        im_data_dict["Dimensions"] = dimensions
        im_data_dict["PixelDepth"] = self.depth
        im_data_dict["break"] = True  # temdm tag
        im_data_dict["Dimensions"]["sorted"] = False

        im_dict = {"ImageData": im_data_dict}

        if metadata is not None:
            im_dict["ImageTags"] = metadata

        im_list_dict = {"ImageList": {"0": im_dict}}
        im_list_dict["ImageList"]["break"] = True  # temdm tag
        im_list_dict["ImageList"]["sorted"] = False  # temdm tag
        im_list_dict["ImageList"]["0"]["break"] = True  # temdm tag

        if dim == 1:
            Dtype = 13
        else:
            Dtype = 1
        obj_list_dict = {
            "DocumentObjectList": {
                "0": {
                    "AnnotationType": 20,
                    "ImageDisplayType": Dtype,
                    "ImageSource": 0,
                    "UniqueID": 8,
                },
                "sorted": False,
            }
        }  # temdm tag

        im_beh_dict = {"Image Behavior": {"ViewDisplayID": 8}}

        im_source_dict = {
            "ImageSourceList": {
                "0": {"ClassName": "ImageSource:Simple", "Id": {"0": 0}, "ImageRef": 0}
            }
        }
        im_source_dict["ImageSourceList"]["sorted"] = False  # temdm tag
        im_source_dict["ImageSourceList"]["0"]["Id"]["sorted"] = False  # temdm tag

        mode_dict = {"InImageMode": 1}

        self.tags_dict.update(obj_list_dict)
        self.tags_dict.update(im_beh_dict)
        self.tags_dict.update(im_list_dict)
        self.tags_dict.update(im_source_dict)
        self.tags_dict.update(mode_dict)

        return self.tags_dict

    """
    basic sequence of writting file
    """

    def write_to_buffer(self):
        self.b = bytearray(b"")  # create buffer
        self.write_dict(self.tags_dict, root=True)  # write dictionary to buffer
        self.finalize_buffer()

    def break_buffer(self):
        self.image_tag_header()  # write imgae tag header
        self.b_before = self.b  # store so far written buffer
        self.b = bytearray(b"")  # reset buffer

    def finalize_buffer(self):
        self.b_after = self.b  # store the 2nd part of buffer
        f_size = (
            len(self.b_before) + len(self.b_after) + self.depth * self.size
        )  # deduce file size
        if self.print:
            print("file size", f_size)
        self.b_header = self.file_header(f_size)  # store the file size

    def file_header(self, f_size: int) -> bytes:
        # prepare the file header
        b_header = b""
        b_header += set_int32(self.dm_ext, "big")  # 3(dm3) or 4(dm4)

        b_header += self.set_DMvalue(f_size)  # size of the file

        if self.endian == "little":
            end = 1
        else:
            end = 0
        b_header += set_int32(end, "big")  # set endian

        return b_header

    def group_header(self, is_sorted: int, n_tags: int, root: bool):
        if self.dm_ext == 4 and root is False:
            self.b += set_uint64(0, "big")  # grouplength is set to Zero initially
            self.group_stack.append(len(self.b))  # store offset

        self.b += set_uint8(is_sorted, "big")  # is group sorted
        self.b += set_uint8(0, "big")  # group is not open

        self.b += self.set_DMvalue(n_tags)  # number of tags in group

    def name_header(self, Ttype, Label):
        if Ttype == "dict":
            self.b += set_uint8(20, "big")  # that is a tag group not single tag
        else:
            self.b += set_uint8(21, "big")  # that is a tag

        if Label in tag_counts():
            Label = ""

        self.b += set_uint16(len(Label), "big")  # Label length
        self.b += self.set_string(Label)  # Label of tag or tagGroup

    def set_string(self, TxtLabel: str) -> bytes:
        """
        1byte char (DM code:9) for tag labels
        """
        return TxtLabel.encode("utf-8")

    def tag_header(self, Ttype, item):
        if self.dm_ext == 4:
            self.b += set_uint64(0, "big")  # tag length (set to zero for the moment)
            self.tag_address = len(self.b)  # memorize the address for this note
        self.b += self.set_string("%%%%")  # marker for starting tag

    def image_tag_header(self):  # includes also the image name header
        self.b += set_uint8(21, "big")  # that is a tag, not tag group
        self.b += set_uint16(4, "big")  # Label length
        self.b += self.set_string(
            "Data"
        )  # replace the signal code for the standard image data label

        if self.dm_ext == 4:
            length = self.size * self.depth + 16 + 8 + 8 + 4
            self.b += set_uint64(length, "big")  # length of the image array

        self.b += self.set_string("%%%%")  # marker for starting tag

        self.b += self.set_DMvalue(3)  # 3 positions needed to describe an array
        self.b += self.set_DMvalue(20)  # 20: DM code of array (1st position)
        self.b += self.set_DMvalue(self.type)  # type of array elements (2nd position)
        self.b += self.set_DMvalue(self.size)  # size of array (3th position)
        if self.print:
            print("Image Data headed")

    def tag_end(self):
        """
        this is needed to write the tag length
        """
        offset = self.tag_address  # retrive the last stored address of the tag length
        tag_length = len(self.b) - offset
        tag_length_b = set_int64(tag_length, "big")
        self.b[offset - 8 : offset] = tag_length_b

    def tag(self, Ttype: str, item):
        self.tag_header(Ttype, item)  # write tag header

        if Ttype in NP_number_types():  # a number
            self.b += self.set_DMvalue(1)  # 1 position needed to describe a number type
            self.b += self.set_DMvalue(self.type_dict_NP[Ttype][2])  # DM code
            self.b += self.type_dict_NP[Ttype][4](item, self.endian)  # number self

        elif Ttype == "str":  # a string
            """it seems DM does not read string unless it is converted into arrays of symbols
            self.b +=self.set_DMvalue(2) # 2 positions needed to describe a string
            self.b +=self.set_DMvalue(18) #18 -DM code of string (1st position)
            length =len(item)
            self.b +=self.set_DMvalue(length) #length of string (2nd position)
            self.b +=self.set_string(item) #string self
            """

            self.b += self.set_DMvalue(3)  # 3 positions needed to describe an array
            self.b += self.set_DMvalue(20)  # 20 -DM code of array (1st position)
            self.b += self.set_DMvalue(4)  # type of array elements uint8 (2nd position)
            size = len(item)
            self.b += self.set_DMvalue(size)  # size of array (3th position)
            for i in range(size):
                self.b += set_uint16(ord(item[i]), self.endian)

        elif Ttype == "list" or Ttype == "ndarray":  # an array
            self.b += self.set_DMvalue(3)  # 3 positions needed to describe an array
            self.b += self.set_DMvalue(20)  # 20 -DM code of array (1st position)
            eltype = self.get_type(item[0])  # take type of the first array element
            elcode = self.type_dict_NP[eltype][2]  # DM code
            self.b += self.set_DMvalue(elcode)  # type of array elements (2nd position)
            size = len(item)
            self.b += self.set_DMvalue(size)  # size of array (3th position)
            if elcode == 4:
                for i in range(size):
                    # print(item[i])
                    self.b += self.type_dict_NP[eltype][4](ord(item[i]), self.endian)
            else:
                for i in range(size):
                    # print(item[i])
                    self.b += self.type_dict_NP[eltype][4](
                        item[i], self.endian
                    )  # array self

        elif Ttype == "tuple":  # a tuple
            size = len(item)
            self.b += self.set_DMvalue(
                3 + 2 * size
            )  # 3+2*size positions needed to describe a tuple
            self.b += self.set_DMvalue(15)  # 15 -DM code of string (1st position)
            self.b += self.set_DMvalue(0)  # dummy ZERO (2nd position)
            self.b += self.set_DMvalue(size)  # size of tuple (3rd position)

            eltypes = ()  # tuple of types
            for i in range(size):  # 2*size positions
                self.b += self.set_DMvalue(0)  # dummy ZERO
                eltype = self.get_type(item[i])  # take type of the tuple element
                eltypes += (eltype,)  # tuple of types
                elcode = self.type_dict_NP[eltype][2]  # DM code
                self.b += self.set_DMvalue(elcode)  # type of tuple element
            for i in range(size):
                self.b += self.type_dict_NP[eltypes[i]][4](
                    item[i], self.endian
                )  # element self

        else:  # other tags are not supported
            print("unknown type:", Ttype, "skipped")

        if self.dm_ext == 4:
            self.tag_end()  # write the tag length

    def group_end(self, is_broken: bool):
        offset = (
            self.group_stack.pop()
        )  # retrieve the recent address of the taggroup length

        if offset == (-1):  # dm3 or dm4 root group, no need to store the group length
            self.group_stack.append(-1)  # reset buffer
        else:
            if is_broken is True:
                group_length = (
                    len(self.b_before) - offset + self.size * self.depth + len(self.b)
                )
                group_length_b = set_int64(group_length, "big")
                # as offset was accounted in b_before
                self.b_before[offset - 8 : offset] = group_length_b
            else:
                group_length = len(self.b) - offset
                group_length_b = set_int64(group_length, "big")
                self.b[offset - 8 : offset] = group_length_b

            if self.print:
                print("end of group", "offset", offset, "groupLength", group_length)

    def read_temdm_tag(self, group_dict: dict, name: str, par: bool) -> bool:
        """
        remove temdm special tags from dictionary, overwrite the default parameter if neccesary
        """
        if name in group_dict:
            par = group_dict[name]  # read parameter, overwrite default
            del group_dict[name]  # after reading, the tag is not needed anymore

        return par

    def write_dict(self, group_dict: dict, root=False):
        """
        recursive function to write nested dictionary into binary
        """
        is_sorted = self.read_temdm_tag(group_dict, "sorted", True)  # default sorted
        is_broken = self.read_temdm_tag(group_dict, "break", False)  # default no break

        # write group header
        n_tags = len(group_dict)
        self.group_header(is_sorted, n_tags, root)
        if self.print:
            print("group: tags", n_tags)

        # Iterate over all key-value pairs in dictionary
        for key, value in group_dict.items():
            if key == "DataImageArray":  # temdm special name signalling the image data
                self.break_buffer()
            else:
                Ttype = self.get_type(value)
                if key == "ImageSource":
                    Ttype = "uint64"  # overwrite uint32 to uint64
                if key == "InImageMode":
                    Ttype = "uint8"  # overwrite uint32 to bool

                self.name_header(Ttype, key)  # write the name header
                if self.print:
                    print("   name:", key, Ttype, value)
                if Ttype == "dict":  # nested tagGroup
                    self.write_dict(value)
                else:
                    self.tag(Ttype, value)  # write  a tag

        if self.dm_ext == 4:
            self.group_end(is_broken)  # write the tag group length
