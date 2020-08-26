import os
import time
import pickle
import numpy as np
from PIL import Image
#from numba import vectorize
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def display_image(img):
    plt.imshow(img, cmap="Greys")
    plt.axis('off')
    plt.show()

def find_patches(images, m):
    patches = dict()
    for image in images:        
        patches[image[0]]=[]
        print('.', end='')
        i=0
        while i<len(image[1])-m+1:
            j=0
            while j<len(image[1][0])-m+1:
                sub = image[1][i:i+m, j:j+m]
                #all_zeros = not np.any(sub)
                #if not all_zeros:
                patches[image[0]].append(np.array(sub))
                j+=1
            i+=1
    return patches

def noisfy(corp):
    nimages=[]
    for i in range(len(corp)):    
        nimages.append([corp[i][0], gaussian_filter(corp[i][1], sigma=1.5)])
    print('Pickles made')
    print("Number of noisy images: ", len(nimages))
    return np.array(nimages)
    #images[k][i][j]+=noise_factor*np.random.normal(loc=0.0, scale=1.0, size=1) 

def to_binary(images):
    bin_images = dict()
    for item in images:
        bin_images[item]=[]
        print('=', end='')
        for j in range(len(images[item])):
            image = []
            for k in range(len(images[item][j])):
                dat = np.divide(images[item][j][k], 255)
                dat = np.ceil(dat)
                #dat = np.floor(dat)
                dat = dat*255
                image.append(dat)
            bin_images[item].append(np.array(image))
    return bin_images
            
def lis(images):
    res = set()
    for item in images:
        for patch in images[item]:
            #print(tuple(map(tuple, patch)))
            res.add(tuple(map(tuple, patch)))
    return list(res)

def load_data(path):
    images=[]
    for folder in os.listdir(path):
        folder_path = path+folder
        print(folder_path)
        for filename in os.listdir(folder_path):
            cur_path = folder_path+"\\"+filename
            im = np.array(Image.open(cur_path))
            #im = np.divide(im, 255)
            images.append([filename, im])
    print("Number of images: ", len(images))
    return images

def display_set(images, cut=10):
    print('--------------------------------------------')
    j=0
    for item in images:
        j=0
        for patch in images[item]:
            all_zeros = not np.any(patch)
            if not all_zeros:
                display_image(patch)
                j+=1
            if j==cut:
                break
        break
    print('--------------------------------------------')

#@jit(target ="cuda")
def compute_bin_given_gray(bins, grays):
    probabs = dict()
    for im in bins:
        probabs[im] = dict()
        for i in range(len(bins[im])):
            print('*', end='')
            n1 = np.divide(bins[im][i], 255)
            n2 = 1 - n1
            probabs[im][i] = dict()
            for j in range(len(grays[im])):
                n3 = grays[im][j]#np.divide(grays[im][j], 255)
                n4 = 255 - n3
                #res = np.log(np.multiply(n1, n3)) + np.log(np.multiply(n2, n4))
                res = np.multiply(n1, n3) + np.multiply(n2, n4)
                res[res==0] = 1
                #prob = np.sum(res)
                prob = np.prod(res)
                probabs[im][i][j]=prob
    return probabs

# =============================================================================
# def compute_bin_given_gray(bins, grays):
#     probabs = dict()
#     for im in bins:
#         probabs[im] = dict()
#         for i in range(len(bins[im])):
#             print('*', end='')
#             probabs[im][i] = dict()
#             for j in range(len(grays[im])):
#                 if i==j:
#                     probabs[im][i][j]=1
#                 else:
#                     probabs[im][i][j]=0
#     return probabs
# =============================================================================

def compute_likely(probabs_T, probabs_F, pcap_T, bin_lowres, bin_highres):
    probabs_FT = dict()
    for im in probabs_T:
        probabs_FT[im] = dict()
        i=0
        while i < len(bin_lowres[im]) and i<len(bin_highres[im]):
            probabs_FT[im][i] = dict()
            j=0
            while j < len(bin_lowres[im]) and j<len(bin_highres[im]):
                probabs_FT[im][i][j]= probabs_T[im][i][j]*probabs_F[im][i][j]
                probabs_FT[im][i][j]/=(len(bin_highres[im][i]) * pcap_T)
                j+=1
            i+=1
    return probabs_FT

path = 'E:\IIITD\Semester 2\PGM\Project\kkanji\\test\\'
pickle_path = 'E:\IIITD\Semester 2\PGM\Project\savedtest\\'
imgpath = 'E:\IIITD\Semester 2\PGM\Project\\images\\'

high = 5
low = 10
start_time = time.time()  
try:
    tup = pickle.load(open(pickle_path+'SISR', "rb"))
    images, nimages, highres, lowres, bin_highres, bin_lowres = tup
    
    #tup_lis = pickle.load(open(pickle_path+'SISR_lis', "rb"))
    #highs, lows, bin_highs, bin_lows = tup_lis
    print("pickles loaded")
except:
    print("pickles missing")
    images = load_data(path)
    nimages = noisfy(images)
    highres = find_patches(images, high)
    lowres = find_patches(nimages, low)
    bin_highres = to_binary(highres)
    bin_lowres = to_binary(lowres)
#    pickle.dump((images, nimages, highres, lowres, bin_highres, bin_lowres), open(pickle_path+'SISR', "wb"))    
    
    #highs = lis(highres)
    #lows = lis(lowres)
    #bin_highs = lis(bin_highres)
    #bin_lows = lis(bin_lowres)
    #pickle.dump((highs, lows, bin_highs, bin_lows), open(pickle_path+'SISR_lis', "wb"))    

#print(ncorpus[0])
display_image(images[0][1])
display_image(nimages[0][1])

k=214
for im in highres:
    display_image(highres[im][k])
    break
print('--------------------------------------------')
for im in lowres:
    display_image(lowres[im][k])
    break
print('--------------------------------------------')

for im in bin_highres:
    display_image(bin_highres[im][k])
    break
print('--------------------------------------------')
for im in bin_lowres:
    display_image(bin_lowres[im][k])
    break
print('--------------------------------------------')
s = 0
for im in bin_lowres:
    s+=len(bin_lowres[im])
print(s)

s = 0
for im in lowres:
    s+=len(lowres[im])
print(s)


try:
    tup = pickle.load(open(pickle_path+'SISR_pr', "rb"))
    probabs_T, probabs_F = tup
    
    #tup_lis = pickle.load(open(pickle_path+'SISR_lis', "rb"))
    #highs, lows, bin_highs, bin_lows = tup_lis
    print("pickles loaded")
except:

    probabs_T = compute_bin_given_gray(bin_lowres, lowres)
    probabs_F = compute_bin_given_gray(bin_highres, highres)
    
    pcap_T = 0
    for im in probabs_T:
        for k in probabs_T[im]:
            for l in probabs_T[im][k]:
                pcap_T+=np.mean(probabs_T[im][k][l])
    
    probabs_FT = compute_likely(probabs_T, probabs_F, pcap_T, bin_lowres, bin_highres)
#    pickle.dump((probabs_T, probabs_F), open(pickle_path+'SISR_pr', "wb"))
print()
print(len(bin_lowres))
print(len(lowres))
print(len(bin_highres))
print(len(highres))
print(len(probabs_T))
print(len(probabs_F))

for im in highres:
    print(len(bin_lowres[im]))
    print(len(lowres[im]))
    print(len(highres[im]))
    print(len(bin_highres[im]))
    print(len(probabs_T[im]))
    print(len(probabs_F[im]))

# =============================================================================
# img_no = 2
# for im in probabs_T:
#     m = 0
#     for k in probabs_T[im]:
#         if probabs_T[im][(img_no, m)]<probabs_T[im][k]:
#             _, m = k
#     print(probabs_T[im][(img_no,m)])           
#     display_image(lowres[im][m])
#     display_image(bin_lowres[im][m])
# 
# =============================================================================
    
# =============================================================================
# for im in probabs_F:
#     for k in range(20):#probabs_F[im]:
#         abc = sorted(probabs_F[im][k].items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
#         print(abc[0], k)
#         print('High Res Gray:')
#         display_image(highres[im][abc[0][0]])
#         print('High Res Bin:')
#         display_image(bin_highres[im][k])
#         
# =============================================================================

print('RESULTS')

for im in probabs_T:
    for k in range(20):#probabs_F[im]:
        abc = sorted(probabs_T[im][k].items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        print(abc[0], k)
        print('High Res Gray:')
        display_image(lowres[im][abc[0][0]])
        print('High Res Bin:')
        display_image(bin_lowres[im][k])


replacements = dict()
for im in probabs_FT:
    replacements[im]=dict()
    for i in range(len(lowres[im])):
        matches = sorted(probabs_FT[im][i].items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
        #print(i, matches[0])
        replacements[im][i]=matches[0][0]
        
sisr_images = dict()                              
for im in replacements:
    sisr_images[im]=np.zeros((64,64))
    startx = 0
    starty = 0
    end = 64-high+1
    for num in replacements[im]:
        patch = bin_highres[im][num]
        res = sisr_images[im][:]
        res[startx:startx+patch.shape[0],starty:starty+patch.shape[1]] = patch
        #sisr_images[im] = sisr_images[im]+res
        #sisr_images[im][sisr_images[im]>255] = 255
        sisr_images[im] = np.divide(sisr_images[im]+res,2)
        if startx == end-1:
            startx=0
            starty+=1
        if starty == end-1:
            break
        startx+=1        

for im in sisr_images:
    #display_image(images[im][1])        
    display_image(sisr_images[im])        

print("Time: ", time.time()-start_time) 

