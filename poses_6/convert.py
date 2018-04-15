import numpy as np
import math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# Change input filename here
data = np.loadtxt('06.txt', delimiter=' ')
f_list = np.zeros((data.shape[0], 6))
for i in range(data.shape[0]):
    temp_data = data[i,:]
    poses = [temp_data[3], temp_data[7], temp_data[11]]
    poses = np.asarray(poses)
    rot_mat = [[temp_data[0], temp_data[1], temp_data[2]],
               [temp_data[4], temp_data[5], temp_data[6]], [temp_data[8], temp_data[9], temp_data[10]]]
    rot_mat = np.asarray(rot_mat)
    #print(rot_mat)
    eulers = rotationMatrixToEulerAngles(rot_mat)

    f_list[i, :] = np.concatenate((poses, eulers), axis=0)

# Change output filename here 
np.savetxt('06_6_txt', f_list, delimiter=',')




