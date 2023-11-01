import numpy as np
import scipy as sp
from scipy.interpolate import interpn
import KTCFwd
import KTCMeshing
import KTCRegularization
import KTCPlotting
import KTCScoring
import KTCAux
import phantom
import glob
from scipy.stats import norm
import matplotlib.pyplot as plt # Requires pip install cv2 and math


def GenerateMeasurement(input_folder, groundtruth_folder, noise_level = 1, categoryNbr = 1, error_system = (0,0)):
    # Arguments: input_folder (for the reference data, to load current injection matrix, so we know which electrodes to remove...)
        # Noise parameter, distribution type, difficulty level, 
        # error_system: if (0,0), the system error is estimated (runtime is 6 times longer)
        # otherwise, the system error can be supplied.
    # Outputs: The measurement, the phantom image (and perhaps the 
        # conductivity defined on the mesh used in the forward map).
    # Specific settings can be changed below. These include: 
        # Changing the way that phantoms are generated,
        # Changing the conductivity values on the phantom,
        # Other Noise options.
    Nel = 32 # Number of electrodes
    sigmavalues = (0.8,0.001,100)
    additivity = 0.08 # If close to zero: Noise completely scaled by Uelref, if very large: Noise completely normal distributed.
    z = (1e-6) * np.ones((Nel, 1))  # contact impedances
    normalization_norm = 2 # The norm used to normalize data. If = p, and noise_level = 1, the p-norm of the added noise
    # is equal to the average p-norm of the error of the true data. (Currently only works for p=1 and p=2)

    # Future changes: Do forward modelling with FEnics.
    # Considerations on noise: Is there noise on the difference to the reference level,
        # or is the noise in terms of the absolute voltage measurements?
    # Why must input_folder and groundtruth_folder folder be passed as arguments? To create the mesh, and get the reference data
    # groundturth_folder is passed to get the system error.

    # Generate phantom
    sigma_class = phantom.generate(inclusions = 2, # Number of inclusions
                            radius = 0.95, # Radius of the disc (full disc is 1)
                            inter_distance = 20, # Minimum l1 distance between inclusions (in pixels)
                            size = 256, # Image size
                            weights = np.array([4,4,1,1])) # Weights of the different inclusions [rectangles, 
                                                            #ellipses, Voronoi cells, weird blobs]

    # Interpolate to a mesh
    Mesh, Mesh2, vincl, Mpat, Injref = SetupMesh(categoryNbr, Nel,input_folder) # Define the mesh(es), and measurement patterns
    pixwidth = 0.23 / 256
    pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
    pixcenter_y = pixcenter_x
    # Intepolate to ground truth values at mesh points (nearest neighbor and fill with zeros)
    sigma0 = interpn((pixcenter_x,pixcenter_y), np.fliplr(sigma_class.T)*1.0, Mesh.g, method='nearest',bounds_error=False,fill_value=0.0)
    # Change function values to input at different mesh points
    sigma = np.zeros((len(sigma0),1))
    sigma[sigma0<=0.5] = sigmavalues[0]
    sigma[(sigma0<=1.5) * (sigma0>0.5)] = sigmavalues[1]
    sigma[(sigma0<=3.5) * (sigma0>1.5)] = sigmavalues[2]
    
    # Do the forward mapping
    solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)
    Measurement = solver.SolveForward(sigma, z)
    Measurement_nfree = np.array(Measurement) # Noise free measurement
    if len(error_system) == 2: # If default argument, estimate the system error
        error_system = system_error(groundtruth_folder, input_folder,categoryNbr, Nel, average_error = False)

    error_random = norm.rvs(size=Measurement_nfree.shape)
    mat_dict = sp.io.loadmat(input_folder + '/ref.mat') #load the reference data
    Uelref = mat_dict['Uelref']
    error_scaled = error_random/(additivity + abs( Uelref))
    if normalization_norm == 1:
        error_scaled = 55.08 * error_scaled/np.linalg.norm(error_scaled,1)
    else:
        error_scaled = 1.756 * error_scaled/np.linalg.norm(error_scaled,2)

    Measurement_noise = Measurement_nfree + error_system + error_scaled

    sigma_with_values = np.zeros((256,256))
    sigma_with_values[sigma_class < 0.5] = sigmavalues[0]
    sigma_with_values[(sigma_class<=1.5) * (sigma_class>0.5)] = sigmavalues[1]
    sigma_with_values[(sigma_class>1.5)] = sigmavalues[2]
    sigma_class[sigma_class == 3] = 2

    return Measurement_noise, Measurement_nfree, sigma_class, sigma_with_values

def apply_reconstruction(Uel, categoryNbr, Nel, input_folder, Uelref):
    z = 1e-6 * np.ones((Nel, 1))  # contact impedances
    Mesh, Mesh2, vincl, Mpat, Injref = SetupMesh(categoryNbr, Nel,input_folder)
    sigma0 = np.array(0.8*np.ones((len(Mesh.g), 1))) #linearization point
    sigma0_init = sigma0
    corrlength = 1 * 0.115 #used in the prior
    var_sigma = 0.05 ** 2 #prior variance
    mean_sigma = sigma0
    smprior = KTCRegularization.SMPrior(Mesh.g, corrlength, var_sigma, mean_sigma)

    # set up the forward solver for inversion
    solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)

    vincl = vincl.T.flatten()
    # set up the noise model for inversion
    noise_std1 = 0.05;  # standard deviation for first noise component (relative to each voltage measurement)
    noise_std2 = 0.01;  # standard deviation for second noise component (relative to the largest voltage measurement)
    solver.SetInvGamma(noise_std1, noise_std2, Uelref)
    deltaU = Uel - Uelref
    mask = np.array(vincl, bool) # Mask the removed electrodes

    deltaU = Uel - Uelref
    print(deltaU.shape)
    J = solver.Jacobian(sigma0, z)
    deltareco = np.linalg.solve(J.T @ solver.InvGamma_n[np.ix_(mask,mask)] @ J + smprior.L.T @ smprior.L  , J.T @ solver.InvGamma_n[np.ix_(mask,mask)] @ deltaU[vincl])
    sigma = np.array(sigma0 + deltareco)

    return KTCAux.interpolateRecoToPixGrid(sigma, Mesh) # Interpolate to picture


def SetupMesh(categoryNbr, Nel,input_folder):
    # The function sets up mesh, and a matrix specifying which electrodes are left out
    # inputfolder should be a path to where the mesh data is saved. category number specifies
    # the difficulty level
    mat_dict = sp.io.loadmat(input_folder + '/ref.mat') #load the reference data
    Mpat = mat_dict["Mpat"]
    Injref = mat_dict["Injref"] #current injections
    vincl = np.ones(((Nel - 1),76), dtype=bool) #which measurements to include in the inversion
    rmind = np.arange(0,2 * (categoryNbr - 1),1) #electrodes whose data is removed

    #remove measurements according to the difficulty level
    for ii in range(0,75):
        for jj in rmind:
            if Injref[jj,ii]:
                vincl[:,ii] = 0
            vincl[jj,:] = 0

    # load premade finite element mesh (made using Gmsh, exported to Matlab and saved into a .mat file)
    mat_dict_mesh = sp.io.loadmat('Mesh_sparse.mat')
    g = mat_dict_mesh['g'] #node coordinates
    H = mat_dict_mesh['H'] #indices of nodes making up the triangular elements
    elfaces = mat_dict_mesh['elfaces'][0].tolist() #indices of nodes making up the boundary electrodes

    #Element structure
    ElementT = mat_dict_mesh['Element']['Topology'].tolist()
    for k in range(len(ElementT)):
        ElementT[k] = ElementT[k][0].flatten()
    ElementE = mat_dict_mesh['ElementE'].tolist() #marks elements which are next to boundary electrodes
    for k in range(len(ElementE)):
        if len(ElementE[k][0]) > 0:
            ElementE[k] = [ElementE[k][0][0][0], ElementE[k][0][0][1:len(ElementE[k][0][0])]]
        else:
            ElementE[k] = []

    #Node structure
    NodeC = mat_dict_mesh['Node']['Coordinate']
    NodeE = mat_dict_mesh['Node']['ElementConnection'] #marks which elements a node belongs to
    nodes = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC]
    for k in range(NodeC.shape[0]):
        nodes[k].ElementConnection = NodeE[k][0].flatten()
    elements = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT]
    for k in range(len(ElementT)):
        elements[k].Electrode = ElementE[k]

    #2nd order mesh data
    H2 = mat_dict_mesh['H2']
    g2 = mat_dict_mesh['g2']
    elfaces2 = mat_dict_mesh['elfaces2'][0].tolist()
    ElementT2 = mat_dict_mesh['Element2']['Topology']
    ElementT2 = ElementT2.tolist()
    for k in range(len(ElementT2)):
        ElementT2[k] = ElementT2[k][0].flatten()
    ElementE2 = mat_dict_mesh['Element2E']
    ElementE2 = ElementE2.tolist()
    for k in range(len(ElementE2)):
        if len(ElementE2[k][0]) > 0:
            ElementE2[k] = [ElementE2[k][0][0][0], ElementE2[k][0][0][1:len(ElementE2[k][0][0])]]
        else:
            ElementE2[k] = []

    NodeC2 = mat_dict_mesh['Node2']['Coordinate']  # ok
    NodeE2 = mat_dict_mesh['Node2']['ElementConnection']  # ok
    nodes2 = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC2]
    for k in range(NodeC2.shape[0]):
        nodes2[k].ElementConnection = NodeE2[k][0].flatten()
    elements2 = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT2]
    for k in range(len(ElementT2)):
        elements2[k].Electrode = ElementE2[k]

    Mesh = KTCMeshing.Mesh(H,g,elfaces,nodes,elements)
    Mesh2 = KTCMeshing.Mesh(H2,g2,elfaces2,nodes2,elements2)
    return Mesh, Mesh2, vincl, Mpat, Injref

def ForwardFromPhanton(Path, Mesh, Mesh2,Injref, Mpat, vincl,sigmavalues=(0.8,0.8,0.8)):
    # Function computes the forward map from a phantom with 3 regions specified by the sigmavalues.
    # Meshes + injection patterns + matrix specified which nodes are left out, must be provided
    # Path to phantom must be provided.
    Nel = 32  # number of electrodes
    z = (1e-6) * np.ones((Nel, 1))  # contact impedances
    truth = sp.io.loadmat(Path)
    truth = truth["truth"]
    # Find pixel centers
    pixwidth = 0.23 / 256
    pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
    pixcenter_y = pixcenter_x
    # Intepolate to ground truth values at mesh points (nearest neighbor)
    sigma0 = interpn((pixcenter_x,pixcenter_y), np.fliplr(truth.T)*1.0, Mesh.g, method='nearest',bounds_error=False,fill_value=0.0)
    # Change function values to input at different mesh points
    sigma = np.zeros((len(sigma0),1))
    sigma[sigma0<=0.5] = sigmavalues[0]
    sigma[(sigma0<=1.5) * (sigma0>0.5)] = sigmavalues[1]
    sigma[(sigma0<=2.5) * (sigma0>1.5)] = sigmavalues[2]
    # Map to measurement space with forward operator
    solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)
    Measurement = solver.SolveForward(sigma, z)
    Measurement = np.array(Measurement)
    return Measurement

def applyOtsu(sigma, Mesh,sigmavalues = (0.8,0.001,100)):
    # The function takes in a conductivity sigma, defined on Mesh, interpolates to a 256 by 256 image, applies
    # Otsu's  method, and then either returns an image with the classes, or the specifies sigmavalues on the mesh
    # Function returns an array with sigma's dimensions
    # and a 256 by 256 image is returned in the second argument.
    sigma0 = np.array(sigmavalues[0]*np.ones((len(Mesh.g), 1))) # reference conductivity
    deltareco = np.array(sigma - sigma0)
    deltareco_pixgrid = KTCAux.interpolateRecoToPixGrid(deltareco, Mesh) # Interpolate to picture
    level, x = KTCScoring.Otsu2(deltareco_pixgrid.flatten(), 256, 7) # threshold the image histogram using Otsu's method
    deltareco_pixgrid_segmented = np.zeros_like(deltareco_pixgrid)

    # Extract indices corresponding to each class
    ind0 = deltareco_pixgrid < x[level[0]]
    ind1 = np.logical_and(deltareco_pixgrid >= x[level[0]],deltareco_pixgrid <= x[level[1]])
    ind2 = deltareco_pixgrid > x[level[1]]
    inds = [np.count_nonzero(ind0),np.count_nonzero(ind1),np.count_nonzero(ind2)]
    bgclass = inds.index(max(inds)) #background class

    # Set values for the classes
    match bgclass:
        case 0:
            deltareco_pixgrid_segmented[ind1] = 2
            deltareco_pixgrid_segmented[ind2] = 2
        case 1:
            deltareco_pixgrid_segmented[ind0] = 1
            deltareco_pixgrid_segmented[ind2] = 2
        case 2:
            deltareco_pixgrid_segmented[ind0] = 1
            deltareco_pixgrid_segmented[ind1] = 1

    reconstruction = deltareco_pixgrid_segmented

    Nel = 32  # number of electrodes
    z = (1e-6) * np.ones((Nel, 1))  # contact impedances
    # Find pixel centers
    pixwidth = 0.23 / 256
    pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
    pixcenter_y = pixcenter_x
    # Intepolate to ground reconstruction values at mesh points (nearest neighbor)
    sigma0 = interpn((pixcenter_x,pixcenter_y), np.fliplr(reconstruction.T)*1.0, Mesh.g, method='nearest',bounds_error=False,fill_value=0.0)
    # Change function values to input at different mesh points
    sigma_cnst = np.zeros((len(sigma0),1))
    sigma_cnst[sigma0<=0.5] = sigmavalues[0]
    sigma_cnst[(sigma0<=1.5) * (sigma0>0.5)] = sigmavalues[1]
    sigma_cnst[(sigma0<=2.5) * (sigma0>1.5)] = sigmavalues[2]
    return np.array(sigma_cnst), reconstruction

def system_error(GroundTruths_folder, TrainingData_folder,categoryNbr, Nel, average_error = True):
    # Function estimates the systematic error. It calculates mean(U_n - A(sigma_n)) if average_error = True
    # with n ranging over the 5 measurements that we're given. If average_error = False, it calculates the
    # reference error U_ref - A(sigma_ref)
    #TrainingData_folder = 'TrainingData'
    #GroundTruths_folder = 'GroundTruths'
    Path = GroundTruths_folder + '/true' + str(1) + '.mat'
    Mesh, Mesh2, vincl, Mpat, Injref = SetupMesh(categoryNbr, Nel,TrainingData_folder)
    Uelref_nfree = ForwardFromPhanton(Path, Mesh, Mesh2,Injref, Mpat, vincl,(0.8,0.8,0.8))
    mat_dict = sp.io.loadmat(TrainingData_folder + '/ref.mat') #load the reference data
    Uelref = mat_dict['Uelref']
    err = np.squeeze(Uelref - Uelref_nfree)
    error_matrix = np.zeros((len(err), 5))
    error_matrix[:,0] = err
    mat_files = glob.glob(TrainingData_folder + '/data*.mat') # load the real data

    for data_number in range(4):
        Path = 'GroundTruths/true' + str(data_number+1) + '.mat'
        Uel_nfree = ForwardFromPhanton(Path, Mesh, Mesh2,Injref, Mpat, vincl,(0.8,0.001,100))

        # True data
        mat_dict = sp.io.loadmat(mat_files[data_number])
        Uel = mat_dict['Uel']

        err = np.squeeze(Uel - Uel_nfree)
        error_matrix[:,data_number+1] = err

    error_correlation = np.corrcoef(error_matrix.T)
    if average_error:
        error_system = error_matrix.mean(1)
    else:
        error_system = error_matrix[:,0]

    error_system = error_system[:,None]

    return error_system


def ForwardFromPhanton_temp(Path, Mesh, Mesh2,Injref, Mpat, vincl,sigmavalues=(0.8,0.8,0.8),contact_impedance = 1e-6):
    # Function computes the forward map from a phantom with 3 regions specified by the sigmavalues.
    # Meshes + injection patterns + matrix specified which nodes are left out, must be provided
    # Path to phantom must be provided.
    Nel = 32  # number of electrodes
    z = contact_impedance * np.ones((Nel, 1))  # contact impedances
    truth = sp.io.loadmat(Path)
    truth = truth["truth"]
    # Find pixel centers
    pixwidth = 0.23 / 256
    pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
    pixcenter_y = pixcenter_x
    # Intepolate to ground truth values at mesh points (nearest neighbor)
    sigma0 = interpn((pixcenter_x,pixcenter_y), np.fliplr(truth.T)*1.0, Mesh.g, method='nearest',bounds_error=False,fill_value=0.0)
    # Change function values to input at different mesh points
    sigma = np.zeros((len(sigma0),1))
    sigma[sigma0<=0.5] = sigmavalues[0]
    sigma[(sigma0<=1.5) * (sigma0>0.5)] = sigmavalues[1]
    sigma[(sigma0<=2.5) * (sigma0>1.5)] = sigmavalues[2]
    # Map to measurement space with forward operator
    solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)
    Measurement = solver.SolveForward(sigma, z)
    Measurement = np.array(Measurement)
    return Measurement