import tempfile, os, time
import threading
from PIL import Image
#from math import pi,sin,cos,tan,atan2,hypot,floor
import math
from numpy import clip



XFORM_TBL = {}


# given an equalrectangular image of a layer, cuts cubemap tiles at some number of rotations, and appends resulting images to a given zip archive
def cut_tiles_and_package_to_zip(img, layer, panoid, fmt, resize_to=False):
    print("{} {}".format(panoid, layer))
    tiles = _tiles_from_equirectangular(img) # a nested dict of rots and facs

    # resize and prepare filenames
    fns, tis = [],[]
    for rot, faces in tiles.items():
        for fac, img in faces.items():
            if resize_to: img = img.resize((resize_to,resize_to), Image.ANTIALIAS)
            tis.append(img)
            fns.append("{}_{}_{}.{}".format(panoid,rot,fac,fmt))

    
    for fn, ti in zip(fns, tis):
        ti.save(fn) # save img to temp folder

def _tiles_from_equirectangular(img, do_multithread=False):
    # we could alter rotations here if desired
    rots = [0,12,6]  # rot of 12 = 30deg; rot of 6 = 60deg
    ret = {}
    threads = []
    for rot in rots:
        key = '{:02d}'.format(rot)
        img_rot = _rotate_equirectangular(img, rot)
        if do_multithread:
            thrd = threading.Thread(target=_faces_from_equirectangular, args=(img_rot,ret,key))
            thrd.start()
            threads.append( thrd )
        else:
            _faces_from_equirectangular(img_rot,ret,key)

    if do_multithread:
            for i in range(len(threads)): threads[i].join()
    #dur = int(time.clock()-tic)
    #print(ret)
    #if did_calc: print("rotation of {} took {}s and required a calculation".format(rot, dur))
    #else: print("rotation of {} took {}s and required no calculation".format(rot, dur))
    return ret


def _faces_from_equirectangular(img_eqrc, ret, key):
    tic = time.process_time()
    img_cmap = Image.new("RGB",(img_eqrc.size[0],int(img_eqrc.size[0]*3/4)),"black")
    did_calc = _convert_back(img_eqrc,img_cmap)

    dim = face_size(img_eqrc)
    box = (0,0,dim,dim)
    tile_top = Image.new(img_cmap.mode,(dim,dim),color=None)
    tile_top.paste( img_cmap.crop((dim*2,0,dim*3,dim)), box )

    tile_bottom = Image.new(img_cmap.mode,(dim,dim),color=None)
    tile_bottom.paste( img_cmap.crop((dim*2,dim*2,dim*3,dim*3)), box )

    tile_back = Image.new(img_cmap.mode,(dim,dim),color=None)
    tile_back.paste( img_cmap.crop((0,dim,dim,dim*2)), box )

    tile_right = Image.new(img_cmap.mode,(dim,dim),color=None)
    tile_right.paste( img_cmap.crop((dim,dim,dim*2,dim*2)), box )

    tile_front = Image.new(img_cmap.mode,(dim,dim),color=None)
    tile_front.paste( img_cmap.crop((dim*2,dim,dim*3,dim*2)), box )

    tile_left = Image.new(img_cmap.mode,(dim,dim),color=None)
    tile_left.paste( img_cmap.crop((dim*3,dim,dim*4,dim*2)), box )

    ret[key] = {"top":tile_top,"btm":tile_bottom,"bck":tile_back,"rht":tile_right,"fnt":tile_front,"lft":tile_left}
    dur = int(time.process_time()-tic)
    print("face extraction {} took {}s".format(key, dur))
    return True

def face_size(img_eqrc):
    return int(img_eqrc.size[0]/4)

# rotates an equirectangular image
# rot is the amount of rotation, given in terms of integer number of divisions of a circle
# rot=12=30deg; rot=8=45deg; rot=6=60deg; rot=4=90deg
def _rotate_equirectangular(img_src, rot=0):
    if rot==0: return img_src.copy()
    img_tar = Image.new(img_src.mode,img_src.size,color=None)
    fmt = img_src.format
    w,h = img_src.size
    div = int(w/rot) # amount to rotate

    img_tar.paste( img_src.crop((0,0,div,h)), (w-div,0,w,h) )
    img_tar.paste( img_src.crop((div,0,w,h)), (0,0,w-div,h) )
    return img_tar

# adapted from https://gist.github.com/muminoff/25f7a86f28968eb89a4b722e960603fe
# get x,y,z coords from out image pixels coords
# i,j are pixel coords
# face is face number
# edge is edge length
def _out_img_to_xyz(i,j,face,edge):
    a = 2.0*float(i)/edge
    b = 2.0*float(j)/edge
    if face==0: # back
        (x,y,z) = (-1.0, 1.0-a, 3.0 - b)
    elif face==1: # left
        (x,y,z) = (a-3.0, -1.0, 3.0 - b)
    elif face==2: # front
        (x,y,z) = (1.0, a - 5.0, 3.0 - b)
    elif face==3: # right
        (x,y,z) = (7.0-a, 1.0, 3.0 - b)
    elif face==4: # top
        (x,y,z) = (b-1.0, a -5.0, 1.0)
    elif face==5: # bottom
        (x,y,z) = (5.0-b, a-5.0, -1.0)
    return (x,y,z)

# wrote this to speed up trig calculations, but using the lookup table doesn't help
def _xyz_to_params(x,y,z,e, use_table=False):
    if use_table:
        if (x,y,z,e) in XFORM_TBL: return XFORM_TBL[(x,y,z,e)], False

    theta = math.atan2(y,x) # range -pi to pi
    r = math.hypot(x,y)
    phi = math.atan2(z,r) # range -pi/2 to pi/2
    # source img coords
    uf = ( 2.0*e*(theta + math.pi)/math.pi )
    vf = ( 2.0*e * (math.pi/2 - phi)/math.pi)

    ui = math.floor(uf)  # coord of pixel to bottom left
    vi = math.floor(vf)
    u2 = ui+1       # coords of pixel to top right
    v2 = vi+1
    mu = uf-ui      # fraction of way across pixel
    nu = vf-vi

    if use_table: XFORM_TBL[(x,y,z,e)] = (ui,vi,u2,v2,mu,nu)
    return (ui,vi,u2,v2,mu,nu), True

# adapted from https://gist.github.com/muminoff/25f7a86f28968eb89a4b722e960603fe
# convert using an inverse transformation
def _convert_back(imgIn,imgOut):
    inSize = imgIn.size
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    edge = inSize[0]/4   # the length of each edge in pixels
    did_ufvf_calc = False
    for i in range(outSize[0]):
        face = int(i/edge) # 0 - back, 1 - left 2 - front, 3 - right
        if face==2:
            rng = range(0,int(edge*3))
        else:
            rng = range(int(edge), int(edge) * 2)

        for j in rng:
            if j<edge: face2 = 4 # top
            elif j>=2*edge: face2 = 5 # bottom
            else: face2 = face

            (x,y,z) = _out_img_to_xyz(i,j,face2,edge)
            (ui,vi,u2,v2,mu,nu), calced = _xyz_to_params(x,y,z,edge,False)
            if calced: did_ufvf_calc = True

            # Use bilinear interpolation between the four surrounding pixels
            A = inPix[ui % inSize[0],int(clip(vi,0,inSize[1]-1))]
            B = inPix[u2 % inSize[0],int(clip(vi,0,inSize[1]-1))]
            C = inPix[ui % inSize[0],int(clip(v2,0,inSize[1]-1))]
            D = inPix[u2 % inSize[0],int(clip(v2,0,inSize[1]-1))]
            # interpolate
            (r,g,b) = (
              A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
              A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
              A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu )

            outPix[i,j] = (int(round(r)),int(round(g)),int(round(b)))

    return did_ufvf_calc
