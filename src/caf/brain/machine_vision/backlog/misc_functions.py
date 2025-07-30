"""
Created on: 7/23/2025
Original author: Adil Zaheer
"""
# def find_surrounding_names(file_name) -> dict:
#     file_name_list = list(file_name)
#     prefix = file_name_list[:2]
#     prefix = "".join(prefix)
#
#     nums = file_name_list[2:]
#     concat_nums = ''.join(nums)
#     nums_int = int(concat_nums)
#
#     east = nums_int + 100
#     west = nums_int - 100
#     north = nums_int + 1
#     south = nums_int - 1
#     northeast = nums_int + 101
#     northwest = nums_int - 99
#     southeast = nums_int + 99
#     southwest = nums_int - 101
# TODO need to make this more robust to handle crossing boundaries etc. i think this is causing a problem as
# todo some box_boundary names are only 3 numbers
# todo also when it gets to the final new mosaic image it isn't writing out properly? empty image?
#     image_layout_dict = {'centre_image': file_name,
#                          'north': north,
#                          'south': south,
#                          'east': east,
#                          'west': west,
#                          'northeast': northeast,
#                          'southeast': southeast,
#                          'southwest': southwest,
#                          'northwest': northwest}
#
#     for key in image_layout_dict:
#         if key != 'centre_image':
#             image_layout_dict[key] = prefix + str(image_layout_dict[key])
#
#     return image_layout_dict