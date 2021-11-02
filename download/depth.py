import json
import requests
import base64
import numpy as np
import struct
import matplotlib.pyplot as plt


def parse(b64_string):
    # fix the 'inccorrect padding' error. The length of the string needs to be divisible by 4.
    b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
    # convert the URL safe format to regular format.
    data = b64_string.replace("-", "+").replace("_", "/")
    data = base64.b64decode(data)  # decode the string
    # data = zlib.decompress(data)  # decompress the data
    return np.array([d for d in data])


def parseHeader(depthMap):
    return {
        "headerSize": depthMap[0],
        "numberOfPlanes": getUInt16(depthMap, 1),
        "width": getUInt16(depthMap, 3),
        "height": getUInt16(depthMap, 5),
        "offset": getUInt16(depthMap, 7),
    }


def get_bin(a):
    ba = bin(a)[2:]
    return "0" * (8 - len(ba)) + ba


def getUInt16(arr, ind):
    a = arr[ind]
    b = arr[ind + 1]
    return int(get_bin(b) + get_bin(a), 2)


def getFloat32(arr, ind):
    return bin_to_float("".join(get_bin(i) for i in arr[ind: ind + 4][::-1]))


def bin_to_float(binary):
    return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]


def parsePlanes(header, depthMap):
    indices = []
    planes = []
    n = [0, 0, 0]

    for i in range(header["width"] * header["height"]):
        indices.append(depthMap[header["offset"] + i])

    for i in range(header["numberOfPlanes"]):
        byteOffset = header["offset"] + header["width"] * header["height"] + i * 4 * 4
        n = [0, 0, 0]
        n[0] = getFloat32(depthMap, byteOffset)
        n[1] = getFloat32(depthMap, byteOffset + 4)
        n[2] = getFloat32(depthMap, byteOffset + 8)
        d = getFloat32(depthMap, byteOffset + 12)
        planes.append({"n": n, "d": d})

    return {"planes": planes, "indices": indices}


def computeDepthMap(header, indices, planes):

    v = [0, 0, 0]
    w = header["width"]
    h = header["height"]

    depthMap = np.empty(w * h)

    sin_theta = np.empty(h)
    cos_theta = np.empty(h)
    sin_phi = np.empty(w)
    cos_phi = np.empty(w)

    for y in range(h):
        theta = (h - y - 0.5) / h * np.pi
        sin_theta[y] = np.sin(theta)
        cos_theta[y] = np.cos(theta)

    for x in range(w):
        phi = (w - x - 0.5) / w * 2 * np.pi + np.pi / 2
        sin_phi[x] = np.sin(phi)
        cos_phi[x] = np.cos(phi)

    for y in range(h):
        for x in range(w):
            planeIdx = indices[y * w + x]

            v[0] = sin_theta[y] * cos_phi[x]
            v[1] = sin_theta[y] * sin_phi[x]
            v[2] = cos_theta[y]

            if planeIdx > 0:
                plane = planes[planeIdx]
                t = np.abs(
                    plane["d"]
                    / (
                        v[0] * plane["n"][0]
                        + v[1] * plane["n"][1]
                        + v[2] * plane["n"][2]
                    )
                )
                depthMap[y * w + (w - x - 1)] = t
            else:
                depthMap[y * w + (w - x - 1)] = 9999999999999999999.0
    return {"width": w, "height": h, "depthMap": depthMap}


def get_depth_map(pano_id):
    url = "https://www.google.com/maps/photometa/v1?authuser=0&hl=en&gl=uk&pb=!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2suk!3m3!1m2!1e2!2s"
    url += pano_id
    url += "!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3"
    ret = requests.get(url)
    s = json.loads(ret.text[4:])[1][0][5][0][5][1][2]

    depthMapData = parse(s)
    # parse first bytes to describe data
    header = parseHeader(depthMapData)
    # parse bytes into planes of float values
    data = parsePlanes(header, depthMapData)
    # compute position and values of pixels
    return computeDepthMap(header, data["indices"], data["planes"])

def view_depth_map(depthMap):
    # process float 1D array into int 2D array with 255 values
    im = depthMap["depthMap"]
    im[np.where(im == max(im))[0]] = 255
    if min(im) < 0:
        im[np.where(im < 0)[0]] = 0
    im = im.reshape((depthMap["height"], depthMap["width"])).astype(int)
    return im
    # display image
    #plt.imshow(im)
    #plt.show()

from PIL import Image
def depthinfo_to_image(depthinfo, max_d=250.0, pow=0.4, panoid=None, clr_nohit=(255,255,255,0), clr_error=(255,255,255,255) ):
    w, h = depthinfo['width'], depthinfo['height']
    if max(depthinfo['depthMap']) > max_d:
        print("MAXIMUM DEPTH EXCEEDED. {} is greater than {} for panoid {}".format(max(depthinfo['depthMap']),max_d, panoid))
    
    
    # RGBA? TODO
    img = Image.new( "RGBA", (w,h), clr_error )
    pxls = img.load( )
    
    for y in range(h):
        for x in range(w):
            d = depthinfo['depthMap'][y*w + x]
            if d < 0: pxls[x,y] = clr_nohit
            elif d > max_d:
                pass # this pixel will show the error color that the image was initialized with
            else:
                c = int( (d/max_d)**(pow) * 255)
                pxls[x,y] = (c,c,c)
    
    return img
