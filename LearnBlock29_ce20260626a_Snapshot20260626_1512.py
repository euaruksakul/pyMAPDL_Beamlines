from ansys.mapdl.core import launch_mapdl
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# === HELPER FUNCTIONs  =======================================================
# =============================================================================

def parse_sfelist(text_data):
    parsed_records = []
    
    # State tracking variables
    current_element = None
    current_lkey = None
    current_nodes = []
    current_fluxes = []
    
    # Regex to check if a line contains any numbers at all
    has_numbers = re.compile(r'\d')
    
    for line in text_data.strip().split('\n'):
        line = line.strip()
        
        # 1. Skip headers, watermarks, and empty lines
        if not line or "ELEMENT" in line or "*****" in line or "DO NOT USE" in line:
            continue
            
        if not has_numbers.search(line):
            continue
            
        # Split by whitespace
        parts = line.split()
        
        # 2. Case A: This is a NEW element block (line starts with Element ID and LKEY)
        # Check if the line has 4 elements (Elem, Lkey, Node, Flux)
        if len(parts) >= 4:
            # If we were tracking a previous element, save it before starting the new one
            if current_element is not None:
                parsed_records.append({
                    "Element_ID": int(current_element),
                    "LKEY": int(current_lkey),
                    "Face_Nodes": current_nodes,
                    "Heat_Flux": current_fluxes[0] if current_fluxes else 0.0 # Typically uniform per face
                })
            
            # Reset state for the new element block
            current_element = parts[0]
            current_lkey = parts[1]
            current_nodes = [int(parts[2])]
            current_fluxes = [float(parts[3])]
            
        # 3. Case B: This is a CONTINUATION line (only contains Node ID and Flux value)
        elif len(parts) == 2:
            current_nodes.append(int(parts[0]))
            current_fluxes.append(float(parts[1]))
            
    # Don't forget to append the very last element block after the loop finishes
    if current_element is not None:
        parsed_records.append({
            "Element_ID": int(current_element),
            "LKEY": int(current_lkey),
            "Face_Nodes": current_nodes,
            "Heat_Flux": current_fluxes[0] if current_fluxes else 0.0
        })
        
    # Convert into a Pandas DataFrame for analysis
    df = pd.DataFrame(parsed_records)
    return df

# =============================================================================
# =============================================================================
# =============================================================================

mapdl = launch_mapdl()
mapdl.prep7()

# =============================================================================
# ==== Test parameters ========================================================
# =============================================================================
coarse_mesh_size = 0.030 # Mesh size (m)
medium_mesh_size = 0.010
fine_mesh_size = 0.005

dummy_heat_flux = 0 # Dummy heat flux value (W/m^2) to be applied to all surface elements first, which will be overwritten later by the SFE command for the elements hit by the beam.
q_max = 3.5e10  # Peak flux (W/rad^2) - Spectra result is around 35 kW/mrad^2 for 2.0 m IVU

p_source = np.array([10.0, 0.0, 0.0])  # Location of the heat source (m)
p_beam_endpoint = np.array([0.000, 0.0035, 0.0015])  # Endpoint of the beam (m) (this is where the center of the beam is on the sample, to define the beam direction)
beam_direction_vector = p_beam_endpoint - p_source
#NOTE We can add the mis-steer and mis-align here

#Divergence of the gaussian heat load power profile
beam_sigma_H = 0.00030 # Standard deviation of the Gaussian beam in the horizontal direction (rad)
beam_sigma_V = 0.00015 # Standard deviation of the Gaussian beam in the vertical direction (rad)

#Cooling
DIW_convection_coefficient = 10000 # Convection coefficient (W/m^2·K)
DIW_temperature = 25  # Constant temperature in Celsius

# =============================================================================
# ==== Material properties ====================================================
# =============================================================================
# CuCrZr #
mapdl.mp("dens", 1, 8900) # Density (kg/m^3) of CuCrZr  
mapdl.mp("kxx", 1, 330) # Thermal conductivity (W/m·K) of CuCrZr
mapdl.mp("alpx", 1, 1.7e-5) # Coefficient of thermal expansion (1/K) of CuCrZr
mapdl.mp("ex", 1, 128e9) # Young's modulus (Pa) of CuCrZr
mapdl.mp("prxy", 1, 0.33) # Poisson's ratio of CuCrZr
mapdl.mp("c", 1, 385) # Specific heat (J/kg·K) of CuCrZr

# =============================================================================
# ==== SOLID MODELING =========================================================
# =============================================================================
## Main body of the fixed mask
Length = 0.250 # Length of the fixed mask (m)
Width = 0.074 # Width of the fixed mask (m)
Height = 0.074 # Height of the fixed mask (m)

Straight_Hole_Length = 0.045 # Lenght of the downstream straight rectangular hole
Entrance_aperture_Hor = 0.021 # Width of the entrance aperture (m)
Entrance_aperture_Ver = 0.017 # Height of the entrance aperture (m)

Exit_aperture_Hor = 0.005 # Width of the exit aperture (m)
Exit_aperture_Ver = 0.005 # Height of the exit aperture (m)

## Block shape ##
mapdl.block(0, -Length, -Width/2, Width/2, -Height/2, Height/2)

## Wedge shape ##
# k1 = mapdl.k(0,0,-W/2,-H/2)
# k2 = mapdl.k(0,-L/2,W/2,-H/2)
# k3 = mapdl.k(0,-L,W/2,-H/2)
# k4 = mapdl.k(0,-L,-W/2,-H/2)
# area_id = mapdl.a(k1, k2, k3, k4)
# mapdl.vext(area_id, dz=H)

# Define keypoints for the entrance aperture
k0 = mapdl.k("", 0, -Entrance_aperture_Hor/2, Entrance_aperture_Ver/2)
k1 = mapdl.k("", 0, Entrance_aperture_Hor/2, Entrance_aperture_Ver/2)
k2 = mapdl.k("", 0, Entrance_aperture_Hor/2, -Entrance_aperture_Ver/2)
k3 = mapdl.k("", 0, -Entrance_aperture_Hor/2, -Entrance_aperture_Ver/2)

# Define keypoints for the exit aperture
k4 = mapdl.k("", -(Length-Straight_Hole_Length), -Exit_aperture_Hor/2, Exit_aperture_Ver/2)
k5 = mapdl.k("", -(Length-Straight_Hole_Length), Exit_aperture_Hor/2, Exit_aperture_Ver/2)
k6 = mapdl.k("", -(Length-Straight_Hole_Length), Exit_aperture_Hor/2, -Exit_aperture_Ver/2)
k7 = mapdl.k("", -(Length-Straight_Hole_Length), -Exit_aperture_Hor/2, -Exit_aperture_Ver/2)

# Define keypoints for the end of straight hole
k8 = mapdl.k("", -(Length), -Exit_aperture_Hor/2, Exit_aperture_Ver/2)
k9 = mapdl.k("", -(Length), Exit_aperture_Hor/2, Exit_aperture_Ver/2)
k10 = mapdl.k("", -(Length), Exit_aperture_Hor/2, -Exit_aperture_Ver/2)
k11 = mapdl.k("", -(Length), -Exit_aperture_Hor/2, -Exit_aperture_Ver/2)

# mapdl.kplot()

aperture_Volume = mapdl.v(k0, k1, k2, k3, k4, k5, k6, k7)
Straight_Hole_Volume = mapdl.v(k4, k5, k6, k7, k8, k9, k10, k11)

## Cooling channels
Cooling_channel_radius = 0.004 # Radius of the cooling channels (m)
Cooling_channel_V_separation = 0.015 # Vertical separation between cooling channels (m)
Cooling_channel_H_separation = 0.050 # Horizontal separation between left and right cooling channel rows (m)

channel_positions = [
    (-Cooling_channel_H_separation/2, Cooling_channel_V_separation*3/2), (Cooling_channel_H_separation/2, Cooling_channel_V_separation*3/2),
    (-Cooling_channel_H_separation/2, Cooling_channel_V_separation*1/2), (Cooling_channel_H_separation/2, Cooling_channel_V_separation*1/2),
    (-Cooling_channel_H_separation/2, -Cooling_channel_V_separation*1/2), (Cooling_channel_H_separation/2, -Cooling_channel_V_separation*1/2),
    (-Cooling_channel_H_separation/2, -Cooling_channel_V_separation*3/2), (Cooling_channel_H_separation/2, -Cooling_channel_V_separation*3/2)
]

# Rotate the working plane so that the X-axis is along the length of the mask (L direction)
mapdl.wprota(0, 0, -90) 

channel_vols = []
for i, (yc, zc) in enumerate(channel_positions):
    
    # Move the working plane origin to the channel center
    mapdl.wpave(0, yc, zc)
    
    # Create the cylinder: cylind(rad_inner, rad_outer, z_start, z_end)
    v_id = mapdl.cyl4(0,0,Cooling_channel_radius,"","","",Length)
    channel_vols.append(v_id)

# Boolean operation
mapdl.vsbv(1, "ALL") 

## Plot the result
# mapdl.allsel()
# mapdl.vplot(background="w")

# mapdl.asel("S","LOC","Y",-Cooling_channel_H_separation/2-Cooling_channel_radius,-Cooling_channel_H_separation/2+Cooling_channel_radius)
# mapdl.asel("A","LOC","Y",Cooling_channel_H_separation/2-Cooling_channel_radius,Cooling_channel_H_separation/2+Cooling_channel_radius)
# mapdl.aplot(background="w")

# =============================================================================
# ==== MESHING ================================================================
# =============================================================================
print("Meshing the geometry...")

mapdl.allsel()
mapdl.et(1, "SOLID278")
# 1. Set the mesh shape 
mapdl.mshape(1, "3D") # 0 = Octahedral, 1 = Tetrahedral
# 2. Set the meshing key 
mapdl.mshkey(0) # 0 = free, 1 = mapped

# Coarse mesh (overall)
mapdl.esize(coarse_mesh_size) # Element size (m)

#Fine mesh (on the exposed areas and small geometries such as cooling channels)
# mapdl.asel("S", "AREA", "", 6) # Select the area at the entrance (x=0)
# mapdl.asel("S","LOC","X",0) # For Block Shape
# mapdl.asel("S","LOC","X",0,-L/4) # For Wedge Shape

# For Fixed mask shape 
mapdl.asel("S","LOC","Y",-Entrance_aperture_Hor/2,Entrance_aperture_Hor/2)
mapdl.asel("R","LOC","Z",-Entrance_aperture_Ver/2,Entrance_aperture_Ver/2)
mapdl.asel("U","LOC","X",0) # The front side doesn't need to be fine-meshed
mapdl.asel("U","LOC","X",-Length) # The back side doesn't need to be fine-meshed
mapdl.aesize("ALL",medium_mesh_size) # Fine element size (m)

# For cooling channels
mapdl.asel("S","LOC","Y",-Cooling_channel_H_separation/2-Cooling_channel_radius,-Cooling_channel_H_separation/2+Cooling_channel_radius)
mapdl.asel("A","LOC","Y",Cooling_channel_H_separation/2-Cooling_channel_radius,Cooling_channel_H_separation/2+Cooling_channel_radius)
mapdl.aesize("ALL",fine_mesh_size) # Fine element size (m)

mapdl.vmesh("ALL")

# Print total number of elements and nodes
num_elements = mapdl.mesh.n_elem
num_nodes = mapdl.mesh.n_node
print(f"Total number of elements: {num_elements}") # For Student version - total number of elements must be < 128,000
print(f"Total number of nodes: {num_nodes}")

# mapdl.eplot(blocking=False,background="w", show_numbering=True)

# =============================================================================
# ==== DUMMY HEAT FLUX APPLICATION ============================================
# =============================================================================
# Apply dummy heat flux on all external area that may be exposed to the heat load
# To save computation time, we exclude areas that is outside of vacuum too as they won't get any heat from the synchrotron beam. 

mapdl.asel("S", "EXT")
mapdl.asel("R","LOC","Y",-Entrance_aperture_Hor/2,Entrance_aperture_Hor/2) #Only include 'in-vacuum' area
mapdl.asel("R","LOC","Z",-Entrance_aperture_Ver/2,Entrance_aperture_Ver/2)  #Only include 'in-vacuum' area
mapdl.sfa("ALL", 1, "HFLUX", dummy_heat_flux) # Apply a dummy heat flux value 

# mapdl.asel("S", "LOC", "X", 0) # Select the nodes at the entrance (x=0)
# mapdl.sfa("ALL", 1, "HFLUX", dummy_heat_flux) # Apply a dummy heat flux value x2 to the entrance surface elements

# =============================================================================
# ==== CONVECTION APPLICATOIN =================================================
# =============================================================================
# Areas where CONV is applied will also not be included in the HFLUX mapping

mapdl.asel("S","LOC","Y",-Cooling_channel_H_separation/2-Cooling_channel_radius,-Cooling_channel_H_separation/2+Cooling_channel_radius)
mapdl.asel("A","LOC","Y",Cooling_channel_H_separation/2-Cooling_channel_radius,Cooling_channel_H_separation/2+Cooling_channel_radius)

mapdl.sfa("ALL", 1, "CONV", DIW_convection_coefficient, DIW_temperature ) # Cooling with water 

# =============================================================================
# ==== BOUNDARY CONDITIONS ====================================================
# =============================================================================
# mapdl.asel("S", "LOC", "X", -L) # Select the area at the end (x=-L)
# mapdl.nsla("S", 1) # Select the nodes attached to the selected areas

# # Apply constant temperature boundary condition to the selected nodes
# mapdl.d("ALL", "TEMP", DIW_temperature)

# Check the BC assignment
# mapdl.allsel()
# mapdl.eplot(plot_bc=True, plot_bc_legend=True, background="w")

# =============================================================================
# ==== SOLVE (DUMMY heat load)=================================================
# =============================================================================
mapdl.allsel()

mapdl.run("/SOLU")
mapdl.antype("STATIC")
mapdl.solve()
mapdl.finish()

mapdl.allsel()
list_of_surface_elements = mapdl.sfelist("ALL", "HFLUX")
# We use this to generate pd dataframe that contain the list of external faces that may be exposed by the beam

with open("surface_elements_HFLUX.txt", "w", encoding="utf-8") as f:
    f.write(str(list_of_surface_elements))

pd_list_of_surface_elements = parse_sfelist(list_of_surface_elements)
pd_list_of_surface_elements.to_csv("surface_elements_HFLUX_parsed.csv", index=False)
pd_list_of_surface_elements.to_pickle('surface_elements_HFLUX_parsed.pkl') 
#Pickle file (.pkl) export preserves the data types so that we may use the file for later python programming (just in case)

# =============================================================================
# ==== PREPARE FOR LOAD APPLICATIONS ==========================================
# =============================================================================

mapdl.slashsolu() # This is needed for subsequent solves

# Delete the dummy heat flux 
mapdl.allsel()
mapdl.sfadele("ALL", 1, "HFLUX") # Delete the heat flux from all surface elements (we already keep the list of surface elements and their faces)

# =============================================================================
# ==== CONVECTION APPLICATOIN =================================================
# =============================================================================

# mapdl.asel("S", "LOC", "X", -L) # Select the area at the end (x=-L)
# mapdl.sfa("ALL", 1, "CONV", DIW_convection_coefficient, Bulk_temperature ) # Apply a dummy heat flux value x2 to the entrance surface elements

# =============================================================================
# ==== HEAT FLUX APPLICATION ==================================================
# =============================================================================

# Loop through the list of surface elements and apply the actual heat flux value to the elements hit by the beam

normal_vectors = [] # Store the normal vectors for post-processing and visualization
p_volume_centroids = [] # Store the volume centroids for post-processing and visualization
p_face_centroids = [] # Store the face centroids for post-processing and visualization
cos_thetas = [] # Store the dot products for post-processing and visualization

theta_H_array = []
theta_V_array = []
flux_angular_density_array = []
flux_area_density_array = []

hit_element_face_centroid_coord_x = np.array([])
hit_element_face_centroid_coord_y = np.array([])
hit_element_face_centroid_coord_z = np.array([])
hit_element_face_applied_heat_flux = np.array([])

for row in tqdm(pd_list_of_surface_elements.itertuples(), total=len(pd_list_of_surface_elements), desc="Projecting HFLUX"):
    element_id = row.Element_ID
    face_number = row.LKEY
    nodes = row.Face_Nodes
    # Here you would calculate the actual heat flux value based on the beam parameters and the position of the nodes

    # Find the volume centroid of the element
    element_centroid_x = mapdl.get_value(entity="ELEM", entnum=element_id, item1="CENT", it1num="X")
    element_centroid_y = mapdl.get_value(entity="ELEM", entnum=element_id, item1="CENT", it1num="Y")
    element_centroid_z = mapdl.get_value(entity="ELEM", entnum=element_id, item1="CENT", it1num="Z")
    p_volume_centroid = np.array([element_centroid_x, element_centroid_y, element_centroid_z])
    p_volume_centroids.append(p_volume_centroid)

    # Find the coordinates of the four nodes of the face (for octagonal elements, it has to be four nodes, but for tetrahedral elements, it has to be three nodes)
    mapdl.nsel("S", "NODE", "", nodes[0])  # Select the first node of the face
    p0 = np.array(mapdl.mesh.nodes[0])
    mapdl.nsel("S", "NODE", "", nodes[1])  # Select the second node of the face
    p1 = np.array(mapdl.mesh.nodes[0])
    mapdl.nsel("S", "NODE", "", nodes[2])  # Select the third node of the face
    p2 = np.array(mapdl.mesh.nodes[0])
    mapdl.nsel("S", "NODE", "", nodes[3])  # Select the fourth node of the face (for octagonal elements, but for tetrahedral elements, it has to be three nodes)
    p3 = np.array(mapdl.mesh.nodes[0]) # This is not used for finding the normal vector, but it is used for finding the face centroid for octagonal elements. For tetrahedral elements, this will be the same as one of the other three nodes.
    
    mapdl.nsel("S", "NODE", "", nodes[0])
    for node in nodes:
        mapdl.nsel("A", "NODE", "", node) #add all the face nodes into the selection
    points = np.array(mapdl.mesh.nodes) #Create a matrix containing node coordinates
    points = np.unique(points,axis=0) #Remove duplicated nodes (especially for tetragonal mesh)
    
    #Generate the 'bounding box' to calculate the average value of the power profile
    mins = np.min(points, axis=0)  # Contains [x_min, y_min, z_min]
    maxs = np.max(points, axis=0)  # Contains [x_max, y_max, z_max]
    y_min, z_min = mins[1], mins[2]
    y_max, z_max = maxs[1], maxs[2]
    #TO BE USED INSIDE THE IF (cos_theta < 0)..
        
    # face centroid (this has to be modified for octagonal elements later)
    # p_face_centroid = np.mean([p0, p1, p2, p3], axis=0)
    p_face_centroid = np.mean(np.unique([p0, p1, p2, p3], axis=0), axis=0)
    ##NOTE: FOR COMPLEX GEOMETRIES, THERE MAY BE MORE THAN FOUR NODES ON A FACE. KEEP THIS FOR FUTURE REFERENCES.
      
    p_face_centroids.append(p_face_centroid)

    # find unit normal vector of the face.
    v1 = p1 - p0
    v2 = p2 - p0
    normal_vector = np.cross(v1, v2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize

    centroid_to_face_vector = p_face_centroid - p_volume_centroid
    if np.dot(centroid_to_face_vector, normal_vector) < 0:
        normal_vector = -normal_vector  # Flip the normal vector if it's pointing inward    
    normal_vectors.append(normal_vector)

    # ray direction vector (from the heat source to the face centroid)
    ray_direction_vector = p_face_centroid - p_source

    # Calculate the factor based on the angle between the beam direction and the face normal
    cos_theta = np.dot(ray_direction_vector, normal_vector) / (np.linalg.norm(ray_direction_vector) * np.linalg.norm(normal_vector))
    cos_thetas.append(cos_theta)
    # Apply the heat flux only if the face is exposed to the beam
    if (cos_theta) < 0:
        hit_element_face_centroid_coord_x = np.append(hit_element_face_centroid_coord_x, p_face_centroid[0])
        hit_element_face_centroid_coord_y = np.append(hit_element_face_centroid_coord_y, p_face_centroid[1])
        hit_element_face_centroid_coord_z = np.append(hit_element_face_centroid_coord_z, p_face_centroid[2])

        theta_H = np.arctan((p_face_centroid[1]-p_beam_endpoint[1])/(p_face_centroid[0]-p_source[0])) # Horizontal angle (rad)
        theta_V = np.arctan((p_face_centroid[2]-p_beam_endpoint[2])/(p_face_centroid[0]-p_source[0])) # Vertical angle (rad)
      
        theta_H_min = np.arctan((y_min - p_beam_endpoint[1])/(p_face_centroid[0] - p_source[0])) # Horizontal angle (rad)
        theta_V_min = np.arctan((z_min - p_beam_endpoint[2])/(p_face_centroid[0] - p_source[0])) # Vertical angle (rad)
        theta_H_max = np.arctan((y_max - p_beam_endpoint[1])/(p_face_centroid[0] - p_source[0])) # Horizontal angle (rad)
        theta_V_max = np.arctan((z_max - p_beam_endpoint[2])/(p_face_centroid[0] - p_source[0])) # Vertical angle (rad)

        num_points = 20
        theta_H_range = np.linspace(theta_H_min, theta_H_max, num_points)
        theta_V_range = np.linspace(theta_V_min, theta_V_max, num_points)
        H, V = np.meshgrid(theta_H_range, theta_V_range)
        BBox_flux_angular_density = q_max * np.exp(-0.5 * ((H/beam_sigma_H)**2 + (V/beam_sigma_V)**2)) # Gaussian beam profile matrix
        Average_flux_angular_density = np.mean(BBox_flux_angular_density)

        flux_area_density = Average_flux_angular_density / (np.linalg.norm(ray_direction_vector)**2) # Convert from angular density to area density (W/m^2)

        if flux_area_density < 1e-5: #very small numbers may create bug in MAPDL
            flux_area_density = 0

        applied_heat_flux = flux_area_density*abs(cos_theta)

        mapdl.sfe(element_id, face_number, "HFLUX", 1, applied_heat_flux)

        theta_H_array.append(theta_H)
        theta_V_array.append(theta_V)
        flux_angular_density_array.append(Average_flux_angular_density)
        flux_area_density_array.append(flux_area_density) #before applying the cos_theta factor
        hit_element_face_applied_heat_flux = np.append(hit_element_face_applied_heat_flux, applied_heat_flux) #For 3D scatter plot

print(f"the peak power density is {np.max(flux_angular_density_array)} w/rad^2") 
#From this number we can check how well the averaging algorithm works. For small mesh size, this should be very close to the real q_max 

# Plot the centroids of the faces that got hit by the beam
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")  # Enable the 3D toolkit
# Render the nodes as a 3D scatter plot
ax.scatter(
    hit_element_face_centroid_coord_x, hit_element_face_centroid_coord_y, hit_element_face_centroid_coord_z, 
    s=5, 
    c=hit_element_face_applied_heat_flux, 
    cmap='hot', 
    label="Hit Centroids",
    alpha=0.9
)
# Add custom labels matching your SPS-II global coordinate system
ax.set_title("Matplotlib 3D Selected Nodes Verification", fontsize=12)
ax.set_xlabel("X Axis (Beam Direction) [m]")
ax.set_ylabel("Y Axis (Width) [m]")
ax.set_zlabel("Z Axis (Height) [m]")
# Adjust the camera viewing angle (elevation, azimuth)
ax.view_init(elev=20, azim=45)
# plt.show()

# =============================================================================
# ==== SOLVE AGAIN ============================================================
# =============================================================================
mapdl.allsel()

# mapdl.run("/SOLU")
mapdl.antype("STATIC")
mapdl.solve()
mapdl.finish()

mapdl.allsel()
list_of_exposed_surface_elements = mapdl.sfelist("ALL", "HFLUX")
# print("From sfelist() (after solve)")
# print(list_of_exposed_surface_elements)
with open("exposed_surface_elements_HFLUX.txt", "w", encoding="utf-8") as f:
    f.write(str(list_of_exposed_surface_elements))

pd_list_of_exposed_surface_elements = parse_sfelist(list_of_exposed_surface_elements)

pd_list_of_surface_elements['Normal_Vector'] = normal_vectors
pd_list_of_surface_elements['Volume_Centroid'] = p_volume_centroids
pd_list_of_surface_elements['Face_Centroid'] = p_face_centroids
pd_list_of_surface_elements['Cos_Theta'] = cos_thetas

pd_list_of_exposed_surface_elements['Theta_H'] = theta_H_array
pd_list_of_exposed_surface_elements['Theta_V'] = theta_V_array
pd_list_of_exposed_surface_elements['Flux_Angular_Density'] = flux_angular_density_array
pd_list_of_exposed_surface_elements['Flux_Area_Density'] = flux_area_density_array

pd_list_of_surface_elements.to_csv("surface_elements_HFLUX_parsed.csv", index=False)
pd_list_of_exposed_surface_elements.to_csv("exposed_surface_elements_HFLUX_parsed.csv", index=False)

# =============================================================================
# ==== POST-PROCESSING ========================================================
# =============================================================================
mapdl.post1()

# plot temperature distribution
mapdl.set(1) # Set to the first load step

# Plot temperature in an interactive 3D VTK window
mapdl.post_processing.plot_nodal_temperature(background="w", show_edges=True)


# =============================================================================
# ==== STRUCTURAL ANALYSIS (SEQUENTIAL COUPLING) ==============================
# =============================================================================
mapdl.prep7()

mapdl.allsel()

# This automatically changes your 10-node SOLID278 thermal tets 
# into 10-node SOLID187 structural tets!
mapdl.etchg("TTS")

# CRITICAL: Define the "Strain-Free" Reference Temperature.
# This is the temperature at which the mask has zero thermal stress 
# (e.g., room temperature or your cooling water temperature).
mapdl.tref(DIW_temperature)

# You MUST constrain the model, otherwise it will fly away into space when it expands.
# Example: Fixing the back face of the mask (assuming length is 0.250m)
mapdl.allsel()
# mapdl.asel("S", "LOC", "Y", -W/2)
print(f"Width Value (W): {Width} mm")
print(f"Height value (H): {Height} mm")
mapdl.asel("S","LOC","Z",-Height/2)
mapdl.nsla("S", 1)
mapdl.d("ALL", "ALL") # Lock all Degrees of Freedom (UX, UY, UZ) to 0

mapdl.allsel()
mapdl.eplot(plot_bc=True, plot_bc_legend=True, background="w")

# LDREAD reads the temperatures from the previous thermal solve (.rth file) 
# and applies them directly to the nodes as a structural expansion load.
# Print the hidden folder where MAPDL is saving your .rth and .rst files
print(f"MAPDL Working Directory: {mapdl.directory}")
print(f"MAPDL Jobname: {mapdl.jobname}")

mapdl.ldread("TEMP", 1, 1, "", kimg=0, fname=mapdl.jobname, ext="rth")

# =============================================================================
# ==== SOLVE - STRUCTURAL =====================================================
# =============================================================================

mapdl.run("/SOLU")
mapdl.antype("STATIC")
mapdl.solve()
mapdl.finish()

# =============================================================================
# ==== POST PROCESSING - STRUCTURAL ===========================================
# =============================================================================

mapdl.post1()
mapdl.set(1)

print("Plotting Total Thermal Deformation...")
# Plot Total Deformation (Displacement magnitude)
mapdl.post_processing.plot_nodal_displacement(
    "NORM",
    cmap="jet",
    show_edges=True,
    title="Total Thermal Deformation [m]",
    background="w"
)

print("Plotting Von Mises Equivalent Stress...")
# Plot Von Mises Stress
mapdl.post_processing.plot_nodal_eqv_stress(
    cmap="jet",
    show_edges=True,
    title="Von Mises Equivalent Stress [Pa]",
    background="w"
)

# =============================================================================
# ==== STOP MAPDL =============================================================
# =============================================================================
mapdl.exit()